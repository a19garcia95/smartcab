from __future__ import print_function
import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
   

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     
        self.planner = RoutePlanner(self.env, self)  
        self.valid_actions = self.env.valid_actions  
        self.learning = learning 
        self.Q = dict()          
        self.epsilon = epsilon   
        self.alpha = alpha      
        self.t = 0 


    def reset(self, destination=None, testing=False):   
        self.planner.route_to(destination)
        self.a = 0.01 
        self.t += 1
        if testing:
            self.epsilon = 0
            self.alpha = 0
        else:
            self.epsilon = math.exp(-(self.a*self.t)) 

        return None

    def build_state(self):
       
        waypoint = self.planner.next_waypoint() 
        inputs = self.env.sense(self)           
        deadline = self.env.get_deadline(self) 
        state = (waypoint, inputs['light'], inputs['left'], inputs['right'], inputs['oncoming'])
        if self.learning:
            self.createQ(state)
            
        return state


    def get_maxQ(self, state):
        maxQ = max(self.Q[state].values())

        return maxQ 


    def createQ(self, state):
        if self.learning and state not in self.Q:
            self.Q[state] = {action : 0.0 for action in self.valid_actions}

        return


    def choose_action(self, state):
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        if self.learning:
            if self.epsilon > random.random():
                action = random.choice(self.valid_actions)
            else:
                action = random.choice([action for action, value in self.Q[state].items() if value == self.get_maxQ(state)])
        else:
            action = random.choice(self.valid_actions)
        return action


    def learn(self, state, action, reward):
        if self.learning:
            self.Q[state][action] = self.alpha*reward + (1 - self.alpha)*self.Q[state][action]

        return


    def update(self):
 
        state = self.build_state()          
        self.createQ(state)                 
        action = self.choose_action(state)  
        reward = self.env.act(self, action) 
        self.learn(state, action, reward)   

        return
        

def run():
   
    env = Environment()
    agent = env.create_agent(LearningAgent, learning = True, epsilon = 1, alpha = 0.3)
    env.set_primary_agent(agent, enforce_deadline=True)#
    sim = Simulator(env, update_delay=0.01, display = True, log_metrics=True, optimized = True)
    sim.run(n_test = 10, tolerance = 0.005)


if __name__ == '__main__':
    run()
