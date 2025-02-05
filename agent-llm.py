from together import Together
import numpy as np
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import networkx as nx

API_KEY = "9ff3bef58eab2a12173a3cbb498e69b405f187177aaf2efecbef5b5290b4f37b"
# API_KEY = "7498540083c198b4080c6104cfeaee5cd0634e37e5dd434994898c6ec83a68fe"
class LLMAgent:
    def __init__(self, api_key, init_prompt="", model="meta-llama/Llama-3.3-70B-Instruct-Turbo", agent_index=None):
        """
        Initialize the LLM Agent with API key and default configuration.

        :param api_key: API key for Together API
        :param model: Default LLM model to use
        """
        self.client = Together(api_key=api_key)
        self.model = model
        self.memory = []  # Memory to store interaction history
        self.config = {
            "max_tokens": 300,
            "temperature": 0.8,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1,
            "stop": ["<|eot_id|>", "<|eom_id|>"],
            "stream": True
        }
        self.init_prompt = init_prompt
        self.add_to_memory({"role": "user", "content":  init_prompt})
        self.answer_list = []
        self.tidy_answer_list = []
        self.agent_index = agent_index
       
        

    def set_config(self, **kwargs):
        """
        Update the agent's configuration.

        :param kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)

    def add_to_memory(self, message):
        """
        Add a message to the agent's memory.

        :param message: Message to add (dictionary with 'role' and 'content')
        """
        if isinstance(message, dict) and 'role' in message and 'content' in message:
            self.memory.append(message)
        else:
            raise ValueError("Message must be a dictionary with 'role' and 'content'.")

    def clear_memory(self):
        """
        Clear the agent's memory.
        """
        init_memory = self.memory[0]
        self.memory = []
        self.memory.append(init_memory)

    def chat(self, message):
        """
        Interact with the LLM using the Together API.

        :param message: User message (string)
        :return: Model's response (string)
        """
        self.add_to_memory({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.memory,
            **self.config
        )

        model_response = ""
        for token in response:
            if hasattr(token, 'choices'):
                content = token.choices[0].delta.content
                # print(content, end='', flush=True)  # Print streamed tokens
                model_response += content

        self.add_to_memory({"role": "assistant", "content": model_response})
        return model_response

    def chat_with_agent(self, other_agent, initial_message = ""):
        """
        Enable this agent to chat with another agent.

        :param other_agent: Another instance of LLMAgent
        :param initial_message: Initial message to start the conversation
        :return: Conversation history
        """
        conversation_history = []

        # Start the conversation
        self_prompt = initial_message
        self_prompt = "Based on your initial question, communicate with another agent about the initial question and your original result. Follow the such form: Regarding to the question XXX, my response is XXX, what about you?"
        
        count = 0
        while count < 1:
            count += 1
            
            # Original prompt to agent1 to obtain initial question.
            conversation_history.append({"role": "init", "content": self_prompt})
            agent_1_response = self.chat(self_prompt)
            conversation_history.append({"role": "agent_1", "content": agent_1_response})

            # Agent 2 responds
            agent_2_response = other_agent.chat(agent_1_response)
            conversation_history.append({"role": "agent_2", "content": agent_2_response})

            # Agent 1 responds
            agent_1_response_response = self.chat(agent_2_response)
            conversation_history.append({"role": "agent_1", "content": agent_1_response_response})


        return conversation_history
    
    def check_answer(self, initial_question=""):
        answer = self.chat(initial_question)
        self.answer_list.append(answer)
        
        return answer


class LLMNetwork:
    def __init__(self, api_key, num_agent=10, init_prompt="", model="meta-llama/Llama-3.3-70B-Instruct-Turbo", mode="erdos-Gnp"):
        
        self.adj = self.create_adjacency_matrix(num_agent, 20, mode)
        self.agent_list = self.init_agent_list(model, num_agent, 0.7)
        
        check_prompt = "You will be given an opinion illustrated by someone about his reasoning on who is the USA president now, help me to summary his final answer. You should only output the final answer: trump, biden, or he doesn't know. "
        self.tidy_agent = LLMAgent(api_key=api_key, init_prompt=check_prompt, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K")
        
    def init_agent_list(self, model, num_agent=10, correct_prob=0.7):
        
        # model="meta-llama/Llama-3.3-70B-Instruct-Turbo"
        api_key = API_KEY
        
        ip = "You are given the question: who is the president of the USA as of now? You need to consider the opinion of your neighbours and make a decision. "
        it1 = "The year is 2025 and trump is elected president.  "
        it2 = "Joe Biden is still the president"
        
        ipc = ip+it1
        ipic = ip+it2
        
        init_agent_list = []
        
        # rand = random.random()
        
        # while rand >= 0.9 or rand <= 0.1:
        #     rand = random.random()
        
        true_cnt = round(num_agent*correct_prob)
        false_cnt = num_agent - true_cnt
        cnt_list = [True] * true_cnt + [False] * false_cnt
        random.shuffle(cnt_list)
        
                    
        for i in range(num_agent):
            
            rand = random.random()
            
            if cnt_list[i] == True:
                init_agent_list.append(LLMAgent(api_key=api_key, init_prompt=ipc, model=model, agent_index=i))
            else:
                init_agent_list.append(LLMAgent(api_key=api_key, init_prompt=ipic, model=model, agent_index=i))
                
        return init_agent_list
    
    def check_all_agents(self):
        iq = "Who is the president of USA NOW?"
        
        for i in range(len(self.agent_list)):
            agent = self.agent_list[i]
            agent.check_answer(iq)
            self.agent_list[i] = agent
        
        return
    
    def one_round_comm(self, check=True):
        
        for i in range(len(self.agent_list)):

            center_agent_index = i
            query_agents_index = self.get_attached_agents(i)
            
            self.query_all_agents(center_agent_index, query_agents_index)
            
        if check:
            self.check_all_agents()
        
        return
    
    def multi_round_comm(self, num_round):
        self.check_all_agents()
        for i in range(num_round):
            self.one_round_comm()
        self.tidy_answer_list()
        
        return
    
    def query_all_agents(self, center_agent_index, query_agents_index):
        
        center_agent = self.agent_list[center_agent_index]
        for agent_index in query_agents_index:
            query_agent = self.agent_list[agent_index]
            conversation = center_agent.chat_with_agent(other_agent=query_agent)
            
            self.agent_list[agent_index] = query_agent
            
        return
    
    def multi_round_gather(self, num_round):
    
        self.check_all_agents()
        for i in range(num_round):
            self.one_round_gather()
            
        return
    
    def one_round_gather(self, para=True):
        if not para:
            for i in tqdm(range(len(self.agent_list))):
                center_agent_index = i
                query_agents_index = self.get_attached_agents(i)
                self.gather_from_agents(center_agent_index, query_agents_index)
        else:
            with ThreadPoolExecutor() as executor:
                futures = []
                for i in tqdm(range(len(self.agent_list))):
                    center_agent_index = i
                    query_agents_index = self.get_attached_agents(i)
                    futures.append(executor.submit(self.gather_from_agents, center_agent_index, query_agents_index))

                for future in futures:
                    future.result()
        return
    
    def gather_from_agents(self, center_agent_index, query_agents_index):
        
        p0 = "You are agent with index " + str(center_agent_index) + ". "
        p1 = "You are given the question: who is the president of the USA as of now? You need to consider the opinion of your neighbours and make a decision. "

        p2 = "Your answer was " + self.agent_list[center_agent_index].answer_list[-1] + ". "
        
        pn = ""
        for agent_index in query_agents_index:
            pn += "Agent with index " + str(agent_index) + " has answer: " + self.agent_list[agent_index].answer_list[-1] + ". "
        pq = "Based on these givings, Who is the president of USA NOW? "
        pp = "You should give the answer first, then analyize the reason."
        
        p = p0 + p1 + p2 + pn + pq
        
        center_agent = self.agent_list[center_agent_index]
        center_agent.clear_memory()
        center_agent_response = center_agent.chat(p)
        center_agent.answer_list.append(center_agent_response)
        self.agent_list[center_agent_index] = center_agent

        return center_agent_response
    
    
    def create_adjacency_matrix(self, n: int, k: float, mode="erdos-Gnp") -> np.ndarray:
        """
        Create an adjacency matrix for n agents where each agent has a k% probability 
        of being attached to each of the other agents.
        
        :param n: Number of agents
        :param k: Probability (as a percentage) of connection between agents (0 to 100)
        :return: nxn adjacency matrix (numpy array)
        """
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
            
            values = np.array([i**-r for i in range(1, n+1)], dtype=np.float32)
            distribution = values / np.sum(values)
            cumulative_distribution = np.cumsum(distribution)
            random_value = np.random.rand()
            position = np.searchsorted(cumulative_distribution, random_value)
            
            if position < 99:
                position += 1
        
            for i in range(n):
                possible_edges = [(i, j) for j in range(n)]
                selected_edges = random.sample(possible_edges, position)
                
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
    
    def get_attached_agents(self, agent_index: int) -> list:
        return list(np.where(self.adj[agent_index] == 1)[0])
    
    def get_tidy_answer_list(self):
        
        # self.tidy_answer_list = self.agent_list
        
        for i in range(len(self.agent_list)):
            
            self.agent_list[i].tidy_answer_list = self.agent_list[i].answer_list
            
            for j in range(len(self.agent_list[i].answer_list)):
                
                tidy_response = self.tidy_agent.chat(self.agent_list[i].answer_list[j])
                self.tidy_agent.clear_memory()
                self.agent_list[i].tidy_answer_list[j] = tidy_response
                # if "biden" in self.agent_list[i].answer_list[j].lower():
                #     self.agent_list[i].tidy_answer_list[j] = "biden"
                # elif "trump" in self.agent_list[i].answer_list[j].lower():
                #     self.agent_list[i].tidy_answer_list[j] = "trump"
                # else:
                #     self.agent_list[i].tidy_answer_list[j] = "idk"
        return

def plot_graph(agentopi, figpath):
    
    # 1.2 dollar for llama 70B
    # 
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
    
if __name__ == "__main__":
    network = LLMNetwork(api_key=API_KEY, num_agent=100, mode="perfer-attach", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    # network = LLMNetwork(api_key=API_KEY, num_agent=10, mode="perfer-attach", model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K")

    print("Finish Init")
    network.multi_round_gather(3)
    # print(network.adj)

    print("----------------")
    for agent in network.agent_list:
        print(agent.answer_list)
        
    print("----------------")
    network.get_tidy_answer_list()
    for agent in network.agent_list:
        print(agent.tidy_answer_list)
        
    # print("----------------")
    # for agent in network.agent_list:
    #     print(agent.memory)

    network.get_tidy_answer_list()
    agentopi = np.zeros((len(network.agent_list), len(network.agent_list[0].tidy_answer_list)))
    for i,agent in enumerate(network.agent_list):
        agentopi[i] = np.array(agent.tidy_answer_list) == 'trump'
        
    agentopi.mean(axis=0)
    
    np.save('distri-2-70B.npy', agentopi)
    figpath = "distri-2-70B.png"
    plot_graph(agentopi.mean(axis=0), figpath)
    
