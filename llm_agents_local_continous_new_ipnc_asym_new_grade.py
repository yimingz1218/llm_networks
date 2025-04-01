import numpy as np
import random
import transformers
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import networkx as nx
import os
from llm_dataset import *
import argparse
import re
import ast


def extract_final_conclusion(input_string):
    match = re.search(r"<final conclusion>\s*\[(.*?)\],\s*\[(.*?)\]", input_string, re.DOTALL)
    if match:
        try:
            list1 = [int(num.strip()) for num in match.group(1).split(',') if num.strip().isdigit()]
            list2 = [int(num.strip()) for num in match.group(2).split(',') if num.strip().isdigit()]
            return list1, list2
        except ValueError:
            return None
    return None

def get_index_dict(lst):
    index_map = {}
    for index, num in enumerate(lst):

        for i in range(num):
            
            if (i+1) not in index_map:
                index_map[i+1] = []
                
            index_map[i+1].append(index)
            
    return index_map

def plot_graph(agentopi, figpath):
    
    data = agentopi
    x = np.arange(len(data))  # Create an array for x-axis

    plt.figure(figsize=(8, 5))
    plt.plot(x, data, marker='o', linestyle='-', color='b', label="Values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Graph of Given Data")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    

class LLMAgent:
    def __init__(self, init_prompt="", agent_index=None, ipnc=False):
        self.memory = []  # Memory to store interaction history
        self.init_prompt = init_prompt
        self.add_to_memory({"role": "system", "content":  init_prompt})
        self.answer_list = []
        self.tidy_answer_list = []
        self.agent_index = agent_index
        self.confidence_grade = 6 # init grade
        self.ipnc = ipnc
        self.grade_pad = None
        self.grade_index_pad = None
        
        pg = "Evaluate up to five of your neighbors' responses and assign each a reliability score from 0 to 10. Do not evaluate your own response. Avoid repeating or duplicating content."
        pg1 = "Begin with a base score of 6 and adjust based on the order in which their responses were given. "
        ie = "If you're confident that your original scenario contains the correct answer, you may grade with higher variance to reflect differences in quality or correctness. "
        ie2 = "If you're uncertain, be cautious, even if some agents have answers that differ from yours or from the majority. "
        pg2 = "Seperate your conclusion sentence with <final conclusion>. The final conclusion should be a list of agent index you evaluate in [], and a list of grade in [], both seperated by commas."
        self.grade_prompt = pg + pg1 + ie + ie2 + pg2
        self.self_grade = 6
        self.received_neighbors = False

    def add_to_memory(self, message):
        if isinstance(message, dict) and 'role' in message and 'content' in message:
            self.memory.append(message)
        else:
            raise ValueError("Message must be a dictionary with 'role' and 'content'.")

    def clear_memory(self):
        init_memory = self.memory[0]
        self.memory = []
        self.memory.append(init_memory)
        
    def add_ask(self, question):
        self.add_to_memory({"role": "user", "content": question})
        
    def add_answer(self, model_response):
        self.add_to_memory({"role": "assistant", "content": model_response})
        self.answer_list.append(model_response)
        return model_response
    
    def add_grade(self, list_grade_result):
        if list_grade_result is not None:
            self.grade_index_pad = list_grade_result[0]
            self.grade_pad = list_grade_result[1]
        else:
            self.grade_index_pad = [-1]
            self.grade_pad = [-1]

class LLMNetwork:
    def __init__(self, num_agent=10, correct_prob=0.7, 
                 model_path="/cpfs04/shared/sport/yiming/llms/meta-llama/Llama-3.3-70B-Instruct", 
                 mode="erdos-Gnp", save_path=None, degree_r=27, qa_prompt=None, 
                 model=None, tokenizer=None, args=None, 
                 descend_tf=False, ascend_tf=False, adj=None):
        
        self.descend_tf = descend_tf
        self.ascend_tf = ascend_tf
        self.args = args
        self.total_num_input_tokens = 0
        self.total_num_output_tokens = 0
        
        
        self.num_agent = num_agent
        self.correct_prob = correct_prob
        
        if adj is not None:
            self.adj = adj
            
        else:
            self.adj = self.create_adjacency_matrix(num_agent, degree_r, mode)
            
        self.qa_prompt_list = qa_prompt

        self.default_single_prompt = [
            {"role": "system", "content": "Default."},
            {"role": "user", "content": "Default."}
        ]
        self.default_single_grade_prompt = [
            {"role": "system", "content": "Default."},
            {"role": "user", "content": "Default."},
            {"role": "assistant", "content": "Default."},
            {"role": "user", "content": "Default."},
        ]
        
        self.agent_list = self.init_agent_list()
        self.save_path = save_path
        self.query_order = []
        self.visualize_connection()
        
        self.grade_list = []
        for i in range(len(self.agent_list)):
            self.grade_list.append(self.agent_list[i].confidence_grade)
        
        ## Initialization for the model
        # self.model = transformers.AutoModelForCausalLM.from_pretrained(
        #     model_path, torch_dtype=torch.bfloat16, device_map="auto"
        # )
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="cuda",
            batch_size= 128,  # Increase batch size for better GPU utilization
        )
        
    def extract_last_list(self, text):
        match = re.findall(r'\[.*?\]', text, re.DOTALL)
        if match:
            return match[-1]  # Convert string representation of list to actual list
        else:
            print(text)
        return None
    
    def count_tokens(self, prompt):
        
        if isinstance(prompt, list):
        # Convert chat format to a single text input
            formatted_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt])
        else:
            formatted_text = prompt
            
        # Tokenize
        tokenized_output = self.tokenizer(formatted_text, return_tensors="pt")
        
        # Get number of tokens
        num_tokens = tokenized_output["input_ids"].shape[1]
        
        return num_tokens
        
    def visualize_connection(self):
        
        graph_filename = os.path.join(self.save_path, "connectivity.png")
        matrix_filename= os.path.join(self.save_path, "distribution.png")
        
        G = nx.from_numpy_array(self.adj)
        pos = nx.spring_layout(G, k=1.5, iterations=100)
        degrees = dict(G.degree())
        # node_sizes = [v * 50 for v in degrees.values()]
        node_sizes = [max(100, v ) for v in degrees.values()] 
        
        fig, ax = plt.subplots(figsize=(30, 15))
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=4, font_color="black", font_weight="bold", ax=ax)

        plt.axis("off")  # Hide axes
        plt.savefig(graph_filename, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Improved graph saved as {graph_filename}")
        
        print(self.adj.shape)
        row_sums = self.adj.sum(axis=1)
        
        plt.hist(row_sums, bins=np.arange(row_sums.min(), row_sums.max() + 2) - 0.5, edgecolor='black', alpha=0.7)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Given Data')
        plt.xticks(np.arange(row_sums.min(), row_sums.max() + 1))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
         
        plt.savefig(matrix_filename)
        plt.close()
        
        np.save(os.path.join(self.save_path, "adj.npy"), self.adj)
        
        return

    def init_agent_list(self):
        
        # c_list = ["s1", "s2"]
        # ic_list = ["s3", "s4", "s5"]
        key_c = random.choice(["s1", "s2"])
        key_ic =  random.choice(["s3", "s4", "s5"])
        
        ipc = (
            "Here is the original scenario: " + self.qa_prompt_list[key_c] +
            " Based on this scenario, answer the following question: " + self.qa_prompt_list["iq"] +
            " The scenario may or may not contain the answer. Answer the question accordingly, or say you don't know."
        )

        ipnc = (
            "Here is the original scenario: " + self.qa_prompt_list[key_ic] +
            " Based on this scenario, answer the following question: " + self.qa_prompt_list["iq"] +
            " The scenario may or may not contain the answer. Answer the question accordingly, or say you don't know."
        )


        # ipc = "Here is the original scenario: " + self.qa_prompt_list[key_c] + "and the question: " + self.qa_prompt_list["iq"] + "The scenario may or may not contain the answer, you can either answer or say you don't know. "
        # ipnc = "Here is the original scenario:" + self.qa_prompt_list[key_ic] + "and the question: " + self.qa_prompt_list["iq"] + "The scenario may or may not contain the answer, you can either answer or say you don't know. "
        # ipc = self.qa_prompt_list["ip"] + self.qa_prompt_list["iq"] + self.qa_prompt_list["itc"] # 25%
        # ipic = self.qa_prompt_list["ip"] + self.qa_prompt_list["iq"] + self.qa_prompt_list["itic"] # 25%
        # ipic_nc = self.qa_prompt_list["ipnc"] + self.qa_prompt_list["iq"] + self.qa_prompt_list["itic"] # 25%
        # ipc_nc = self.qa_prompt_list["ipnc"] + self.qa_prompt_list["iq"] + self.qa_prompt_list["itic"] # 25%
        
        init_agent_list = []
        true_cnt = round(self.num_agent * self.correct_prob)
        false_cnt = self.num_agent - true_cnt
        cnt_list = [True] * true_cnt + [False] * false_cnt
        random.shuffle(cnt_list)
        
        rearranged_cnt_list = [False] * len(cnt_list)
        
        if self.args.centrality == "degree":
            print("degree")
            column_sums= self.adj.sum(axis=0)
            sorted_indices = np.argsort(column_sums)[::-1]
            
        elif self.args.centrality == "eigenvalue":
            print("eigenvalue")
            eigenvalues, eigenvectors = np.linalg.eig(self.adj)
            principal_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
            eigenvector_centrality = np.abs(principal_eigenvector) / np.sum(np.abs(principal_eigenvector))
            eigenvector_centrality = np.round(eigenvector_centrality, decimals=8)
            sorted_indices = np.argsort(eigenvector_centrality)[::-1]
            
        elif self.args.centrality == "pagerank":
            print("pagerank")
            d = 0.85
            dim = self.adj.shape[0]
            pr = np.ones(dim) / dim
            out_degree = self.adj.sum(axis=1)
            dangling_nodes = out_degree == 0
            out_degree[dangling_nodes] = 1
            iterations = 100  # Maximum iterations
            tolerance = 1.0e-6
            for _ in range(iterations):
                new_PR = (1 - d) / dim + d * (self.adj.T @ (pr / out_degree))
                if np.linalg.norm(new_PR - pr, 1) < tolerance:  # Check convergence
                    break
                pr = new_PR
            sorted_indices = np.argsort(pr)[::-1]
    
        elif self.args.centrality == "closeness":
            print("closeness")
            G = nx.from_numpy_array(self.adj)
            closeness_centrality = nx.closeness_centrality(G)
            sorted_indices = np.argsort(np.array(list(closeness_centrality.values())))[::-1]
            # Convert to corresponding dictionary keys
            # sorted_keys = np.array([list(data.keys())[i] for i in sorted_indices])
        else:
            rearranged_cnt_list = cnt_list
            
        if self.descend_tf:
            for i in range(true_cnt):
                rearranged_cnt_list[sorted_indices[i]] = True
            # cnt_list = rearranged_cnt_list
        if self.ascend_tf:
            for i in range(true_cnt):
                rearranged_cnt_list[sorted_indices[-i-1]] = True
                
        cnt_list = rearranged_cnt_list
        
        for i in range(self.num_agent):
            if cnt_list[i] == True:
                init_agent_list.append(LLMAgent(init_prompt=ipc,agent_index=i))

            else:
                init_agent_list.append(LLMAgent(init_prompt=ipnc,agent_index=i))

        self.sorted_indices = sorted_indices.tolist()
        
        return init_agent_list
        
    def init_ask(self):
        batched_prompts = []

        for i in range(len(self.agent_list)):
            
            # self.query_order.append(random_selected_index)
            # print(self.sorted_indices)
            order = self.sorted_indices.index(i)
            
            if order <= 30:
                num_words = self.args.output_soft_limit + 10
            elif 30 < order <= 80:
                num_words = self.args.output_soft_limit
            else:
                num_words = min(5, self.args.output_soft_limit-10)
            pl = "(in " + str(num_words) + " words). "
            
        
            single_prompt = copy.deepcopy(self.default_single_prompt)
            single_prompt[0]["content"] = self.agent_list[i].init_prompt
            single_prompt[1]["content"] = self.qa_prompt_list["iq"] + "You should give the answer first, then provide a short reason for your answer " + pl
            
            num_input_tokens = self.count_tokens(single_prompt)
            self.total_num_input_tokens += num_input_tokens
            
            batched_prompts.append(single_prompt)
            
        outputs = self.pipeline(
            batched_prompts,
            max_new_tokens=self.args.output_hard_limit,
            return_full_text=False,
            temperature=0.5
        )
        
        for i, output in enumerate(outputs):
            num_output_tokens = self.count_tokens(output[0]['generated_text'])
            self.total_num_output_tokens += num_output_tokens
        
        self.update_agents(batched_prompts, outputs)
        
        return outputs
    
    def one_round_gather(self):
        batched_prompts = []
        
        for i in range(len(self.agent_list)):
            single_prompt = copy.deepcopy(self.default_single_prompt)
            single_prompt[0]["content"] = self.agent_list[i].init_prompt
            
            long_prompt = self.create_center_agent_prompt(i)

            single_prompt[1]["content"] = long_prompt
            
            batched_prompts.append(single_prompt)
            
        outputs = []
        batch_size = 10
        for i in range(0, len(batched_prompts), batch_size):
            batch = batched_prompts[i:i + batch_size]
            outputs.extend(self.pipeline(batch, max_new_tokens=64, return_full_text=False))

        self.update_agents(batched_prompts, outputs)
        
        return outputs
    
    def one_round_continous_gather(self):
        
        random_selected_index = random.randint(0, 99)
        self.query_order.append(random_selected_index)
        
        # if order <= 30:
        #     num_tokens = 
        
        single_prompt = copy.deepcopy(self.default_single_prompt)
        single_prompt[0]["content"] = self.agent_list[random_selected_index].init_prompt
        
        long_prompt = self.create_center_agent_prompt(random_selected_index)
        
        single_prompt[1]["content"] = long_prompt
        
        output = self.pipeline(single_prompt,
            max_new_tokens=self.args.output_hard_limit,
            return_full_text=False,
        )
        
        input_token_num = self.count_tokens(single_prompt)
        output_token_num = self.count_tokens(output[0]['generated_text'])
        self.total_num_input_tokens += input_token_num
        self.total_num_output_tokens += output_token_num
        
        self.update_single_agent(selected_index=random_selected_index, single_prompt=single_prompt, output=output)
        return output
    
    def one_round_grade(self):
        batched_prompts = []
        
        for i in range(len(self.agent_list)):
            single_grade_prompt =  copy.deepcopy(self.default_single_grade_prompt)
            single_grade_prompt[0]["content"] = self.agent_list[i].init_prompt
            single_grade_prompt[1]["content"] = self.agent_list[i].memory[1]["content"]
            single_grade_prompt[2]["content"] = self.agent_list[i].memory[2]["content"]
            single_grade_prompt[3]["content"] = self.agent_list[i].grade_prompt
            
            batched_prompts.append(single_grade_prompt)
            
        outputs = []
        
        batch_size = 10
        for i in tqdm(range(0, len(batched_prompts), batch_size)):
            batch = batched_prompts[i:i + batch_size]
            outputs.extend(self.pipeline(batch, max_new_tokens=512, return_full_text=False, temperature=0.3))

        self.update_agents_grade(outputs)
        
        return outputs
        
    def update_agents_grade(self, outputs):
        
        for i, output_grade in enumerate(outputs):
            agent = self.agent_list[i]
            list_result = extract_final_conclusion(output_grade[0]['generated_text'])
            agent.add_grade(list_result)
            
        return

    
    def multi_round_gather(self, num_round):
        self.init_ask()
        for i in tqdm(range(num_round)):
            self.one_round_gather()
        return
    
    def multi_round_continous_gather(self, num_round):
        
        self.init_ask()
        for i in tqdm(range(num_round)):
            
            if self.args.trustrank_readjust:
                
                if (i != 0 and i % 170 == 0):
                    self.one_round_grade()
                    self.one_round_update_connection(i)
            
            self.one_round_continous_gather()
            self.record_cost()
        return
    
    def one_round_update_agents_self_grade(self):
        '''
        recalculate self grade for each agent
        recalculate trustrank for each agent
        '''
        return
    
    def one_round_update_connection(self, step):
        '''
        base on the trustrank, update the adj matrix
        '''
        # recalculate the grade
        
        temp_lists = [[] for _ in range(100)]
        
        for i in range(len(self.agent_list)):
            grade_pad = self.agent_list[i].grade_pad
            grade_index_pad = self.agent_list[i].grade_index_pad
            
            if len(grade_pad) != len(grade_index_pad):
                continue
            
            if len(grade_pad) == 0 or len(grade_index_pad) == 0:
                continue
            
            if grade_pad[0] == -1 or grade_index_pad[0] == -1:
                continue
            if max(grade_index_pad) >= len(temp_lists):
                continue
            
            if len(grade_index_pad) >= 3:
                if grade_index_pad[0] == 1 and grade_index_pad[1] == 2 and grade_index_pad[2] == 3:
                    continue
            for j, index in enumerate(grade_index_pad):
                    
                temp_lists[index].append(grade_pad[j])
            
        prev_grade_list = self.grade_list
        self.grade_list = [sum(lst) / len(lst) if len(lst) > 0 else 0 for lst in temp_lists]
        
        print(self.grade_list)
        
        # recalculate the trustrank
        # iterate over 100 agents to update the adj
        coef = 5
        for i in range(len(self.agent_list)):
            
            grade_change = self.grade_list[i] - prev_grade_list[i]
            
            neighbors = np.where(self.adj[i] == 1)[0]
            non_neighbors = np.where(self.adj[i] == 0)[0]
            
            if grade_change <= 0:
                
                delete_count = int(abs(grade_change) * coef)
                if delete_count >= len(neighbors):
                    delete_count = max(0, len(neighbors) - 1)
                    
                if delete_count > 0 and len(neighbors) > 1:
                    to_delete = random.sample(list(neighbors), delete_count)
                    for neighbor in to_delete:
                        self.adj[i, neighbor] = 0
                        # self.adj[neighbor, i] = 0
                
            else:
                non_neighbors = [x for x in non_neighbors if x != i]
                add_count = int(abs(grade_change) * coef)
                
                if add_count > 0 and len(non_neighbors) > 0:
                    to_add = random.sample(non_neighbors, min(add_count, len(non_neighbors)))
                    for neighbor in to_add:
                        self.adj[i, neighbor] = 1
                        # self.adj[neighbor, i] = 1
        
        # save the adj
        np.save(os.path.join(self.save_path, "adj_" + str(step) + ".npy"), self.adj)
        return
    
    def record_cost(self):
        
        cost_save_path = os.path.join(self.save_path, "cost.txt")
        
        with open(cost_save_path, "a", encoding="utf-8") as file:
            file.write(f"Input cost: {self.total_num_input_tokens}\n" )
            file.write(f"Output cost: {self.total_num_output_tokens}\n" )
            
        return
    
    def update_agents(self, batched_prompts, outputs):
        
        # for i in range(len(self.agent_list)):
        for i, output in enumerate(outputs):
            agent = self.agent_list[i]
            agent.clear_memory()
            agent.add_ask(batched_prompts[i][1]["content"])
            agent.add_answer(output[0]['generated_text'])
            self.agent_list[i] = agent
            
        return
    
    def update_single_agent(self, selected_index, single_prompt, output):
        
        agent = self.agent_list[selected_index]
        agent.clear_memory()
        agent.add_ask(single_prompt[1]["content"]) # long prompt
        agent.add_answer(output[0]['generated_text'])
        agent.received_neighbors = True
        self.agent_list[selected_index] = agent
                
        return
    
    def create_center_agent_prompt(self, center_agent_index):
        
        query_agents_index = self.get_attached_agents(center_agent_index)
        
        p0 = "You are a question answering agent with index " + str(center_agent_index) + ". "
        p1 = "You are given the question: " + self.qa_prompt_list["iq"] + "You need to consider the opinion of your neighbours and make a decision. "
        p2 = "Your previous answer to this question was " + self.agent_list[center_agent_index].answer_list[-1] + ". " + "\n"
        pn = "Your neighbouring agents provided the following answers: "
        # for agent_index in query_agents_index:
        #     pn += "Agent with index " + str(agent_index) + " provides an answer: " + self.agent_list[agent_index].answer_list[-1] + "\n"
        for agent_index in random.sample(query_agents_index, min(10, len(query_agents_index))):
            pn += "Agent with index " + str(agent_index) + " provides an answer: " + self.agent_list[agent_index].answer_list[-1] + "\n"
            
        pq = "Based on this set of information, " 
        iq = self.qa_prompt_list["iq"]
        
        ie = "If you're confident that your original scenario contains the correct answer, you should trust your own answer and stick with it. And make sure you indicate it when giving the reason. "
        ie2 = "Otherwise, you should consider the opinions and answers of your neighbors. "
        pp = "You should give the answer first, then provide a short reason for your answer "
        
        order = self.sorted_indices.index(center_agent_index)
        if order <= 30:
            num_words = self.args.output_soft_limit + 10
        elif 30 < order <= 80:
            num_words = self.args.output_soft_limit
        else:
            num_words = min(5, self.args.output_soft_limit-10)
        pl = "(in " + str(num_words) + " words). "
        
        # pg = "Evaluate each at most five of your neighbors' responses and assign a reliability score from 0 to 10. Don't evaluate yourself. Don't output repulicated things."
        # pg1 = "Start with a base score of 6 and adjust based on the order in which their responses were given. "
        # pg2 = "Seperate your conclusion sentence with <final conclusion>. The final conclusion should be a list of agent index you evaluate in [], and a list of grade in [], both seperated by commas."
        
        return p0 + p1 + p2 + pn + pq + iq + ie + ie2 + pp + pl
    
    def get_attached_agents(self, agent_index: int) -> list:
        return list(np.where(self.adj[agent_index] == 1)[0])
    
    def create_adjacency_matrix(self, n: int, k: float, mode="degree-distri") -> np.ndarray:
            # Initialize an nxn matrix with zeros
            adjacency_matrix = np.zeros((n, n), dtype=int)
            
            if mode == "erdos-Gnp":
                # Iterate over the upper triangle of the matrix (excluding the diagonal)
                for i in range(n):
                    for j in range(n):
                        if np.random.rand() < (k / 100):  # Convert k% to a probability
                            adjacency_matrix[i, j] = 1 
                            
            elif mode == "erdos-GnM":
                possible_edges = [(i, j) for i in range(n) for j in range(n)]
                M = int(k * n/100)
                selected_edges = random.sample(possible_edges, M)
                
                for i, j in selected_edges:
                    adjacency_matrix[i, j] = 1 
                
            elif mode == "perfer-attach":
                
                m = int(k*n/100)
                # Generate the init adj matrix
                for i in range(1, m+1):
                    adjacency_matrix[i, 0] = 1
                    
                in_degrees = [m] + [1] * m
                for new_node in range(m+1, n):
                    # Select m existing nodes as targets with probability proportional to their in-degree
                    possible_targets = random.choices(range(new_node), weights=in_degrees, k=m)
                    
                    # Add directed edges from the new node to the selected targets
                    for target in possible_targets:
                        adjacency_matrix[new_node, target] = 1  # Edge from new_node -> target

                    # Update in-degree list
                    in_degrees.append(0)  # New node starts with 0 in-degree
                    for target in possible_targets:
                        in_degrees[target] += 1  # Increase in-degree of target nodes
                        
                pass
            
            elif mode == "degree-distri":
                r = int(k*n/100)
                
                alpha = r/10
                size = self.num_agent
                
                int_arr = (np.random.pareto(alpha, size)*5+1).astype(int)
                
                for i in range(size):
                    possible_edges = [(i, j) for j in range(n) if j != i]
                    selected_edges = random.sample(possible_edges, min(int_arr[i].item(), int(n/2)))
                    # print(len(selected_edges))
                    
                    for a, b in selected_edges:
                        adjacency_matrix[a, b] = 1
            else:
                pass
            # Ensure every agent has at least one attached neighbor
            for i in range(n):
                if not np.any(adjacency_matrix[i]):  # If row i has only zeros (no neighbors)
                    possible_neighbors = list(set(range(n)) - {i})  # All other agents
                    neighbor = np.random.choice(possible_neighbors)  # Randomly select a neighbor
                    adjacency_matrix[i, neighbor] = 1
                    adjacency_matrix[neighbor, i] = 1  # Ensure symmetry

            return adjacency_matrix
    
    def get_tidy_answer_list(self):
        
        # batched_prompts = []
        check_prompt = self.qa_prompt_list["cp"] + " Do NOT add explanations. Respond ONLY with one of the following single values: 1, 0, or -1."

        for i in range(len(self.agent_list)):
            self.agent_list[i].tidy_answer_list = self.agent_list[i].answer_list
        
        list_lens = []
        for i in range(len(self.agent_list)):
            list_lens.append(len(self.agent_list[i].answer_list))
            # for j in tqdm(range(len(self.agent_list[0].answer_list))):
        
        index_dict = get_index_dict(list_lens)
        
        for k,v in index_dict.items():
            
            batched_prompts = []
            
            for index_v in v:
                single_prompt = copy.deepcopy(self.default_single_prompt)
                single_prompt[0]["content"] = check_prompt
                single_prompt[1]["content"] = self.agent_list[index_v].answer_list[k-1]
                batched_prompts.append(single_prompt)
                
            outputs = self.pipeline(
                batched_prompts,
                max_new_tokens=4,
                return_full_text=False,
                temperature=0.1
            )
            
            for index_index_v, index_v in enumerate(v):
                self.agent_list[index_v].tidy_answer_list[k-1] = outputs[index_index_v][0]['generated_text']

        return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run LLM Network')
    parser.add_argument('--degree_r', type=int, default=21, help='Degree r value')
    parser.add_argument('--correct_prob', type=float, default=0.3, help='Correct probability')
    parser.add_argument('--complete_prob', type=float, default=0.8, help='Complete probability')
    
    parser.add_argument('--test_dataset', type=list, required=True, help='Test dataset')
    parser.add_argument('--root_path', type=str, required=True, help='Root path to save results')
    parser.add_argument('--model_name', type=str, required=True, help='')
    
    parser.add_argument('--descend_tf', type=bool, default=False, help='')
    parser.add_argument('--ascend_tf', type=bool, default=False, help='')
    parser.add_argument('--centrality', type=str, default="degree", help='')
    parser.add_argument('--start_partion', type=int, default=0, help='')
    
    parser.add_argument('--output_hard_limit', type=int, default=96, help='')
    parser.add_argument('--output_soft_limit', type=int, default=30, help='')
    parser.add_argument('--check_output_txt_path', type=str, default="long_prompt99.txt", help='')
    parser.add_argument('--trustrank_readjust', action='store_true')
    
    args = parser.parse_args()
    # test_dataset = DATASET
    
    print(args.trustrank_readjust)
    
    test_dataset = DATASET_INCOMPLETE
        
    if args.model_name == "llama_8b":
        model_path = "/mnt/petrelfs/zhangyiming.p/meta-llama/Llama-3.1-8B-Instruct"
    else:
        model_path = "/mnt/petrelfs/zhangyiming.p/meta-llama/Llama-3.3-70B-Instruct"
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    root_path = args.root_path
    
    if os.path.exists(root_path) == False:
        os.mkdir(root_path)
        
    log_path = os.path.join(root_path, args.check_output_txt_path)
    
    with open(log_path, "w", encoding="utf-8") as file:
        file.write("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"  + "\n")
        file.write(f"correct_prob: {args.correct_prob}\n")  # Writing input shape
        file.write(f"complete_prob: {args.complete_prob}\n")  # Writing input shape
        file.write(f"trustrank_readjust: {args.trustrank_readjust}\n")  # Writing input shape
        file.write(f"centrality: {args.centrality}\n")  # Writing input shape
        file.write(f"output_soft_limit: {args.output_soft_limit}\n")  # Writing input shape
        file.write(f"output_hard_limit: {args.output_hard_limit}\n")  # Writing input shape
        file.write("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" + "\n")
    
    for i, data_x in enumerate(test_dataset):
        
        # if i <= args.start_partion:
        #     continue
        
        save_path = os.path.join(root_path, str(i))
        if os.path.exists(save_path) == False:
            os.mkdir(save_path)
        
        
        if i == 0:
            cached_adj = np.load("/mnt/petrelfs/zhangyiming.p/llms/exps_0324/llama8b_imagine_0308_destf_pagerank_asym_limit20/0/adj.npy")
            network = LLMNetwork(num_agent=100, correct_prob=args.correct_prob, mode="degree-distri", 
                             degree_r=args.degree_r, save_path=save_path, qa_prompt=data_x, 
                             model=model, tokenizer=tokenizer,
                             args=args, descend_tf=args.descend_tf, ascend_tf=args.ascend_tf)
            cached_adj = network.adj
            
        else:
            cached_adj = np.load("/mnt/petrelfs/zhangyiming.p/llms/exps_0324/llama8b_imagine_0308_destf_pagerank_asym_limit20/0/adj.npy")
            network = LLMNetwork(num_agent=100, correct_prob=args.correct_prob, mode="degree-distri", 
                             degree_r=args.degree_r, save_path=save_path, qa_prompt=data_x, 
                             model=model, tokenizer=tokenizer,
                             args=args, descend_tf=args.descend_tf, ascend_tf=args.ascend_tf, adj=cached_adj)

        # network.multi_round_gather(10)
        network.multi_round_continous_gather(175)
        ans_path = os.path.join(save_path, "ans.txt")
        
        with open(ans_path, "w") as file:
            for agent in network.agent_list:
                file.write(str(agent.answer_list) + "\n")
                
        network.get_tidy_answer_list()
        
        tans_path = os.path.join(save_path, "tans.txt")
        
        with open(tans_path, "w") as file:
            for agent in network.agent_list:
                file.write(str(agent.tidy_answer_list) + "\n")
                
        qo_path = os.path.join(save_path, "qo.txt")
        with open(qo_path, "w") as file:
            for line in network.query_order:
                file.write(str(line) + "\n")
                
        
