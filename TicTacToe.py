import numpy as np
import math
class TicTacToe:
    def __init__(self,move=None,state=None,to_play = None):
        #print('Starting Game')
        self.result = None
        if state is None:
            self.state = np.resize(np.zeros(9),(3,3))
        else:
            self.state = state
        self.finished = False
        #print(self.state)
        #Ones start
        if to_play is None:
            self.to_play = 1
        else:
            self.to_play = to_play
        
        #Not sure I want to play random mode at init...
#         if move is None:
#             self.play()
#         else:
#             self.play(move=move)
    
    def board(self):
        print(self.state)
    
    def game_status(self):
        self.result = None
        self.finished = False
            
        for i in range(3):
            #If rows have three in row of same number then someone won
            row_result = self.state[i].sum()
            if row_result in (-3,3):
                self.finished=True
                self.result= 'win' if row_result>0 else 'loss'
            #If columns have three in row of same number then someone won
            column_result = np.rot90(self.state,1)[i].sum()
            if column_result in (-3,3):
                self.finished=True
                self.result= 'win' if column_result>0 else 'loss'
        #if either diagonal have three in row of same number then someone won
        diagonal_result = self.state[0][0]+self.state[1][1]+self.state[2][2]
        if diagonal_result in (-3,3):
            self.finished=True
            self.result= 'win' if diagonal_result>0 else 'loss'
        diagonal_result = self.state[0][2]+self.state[1][1]+self.state[2][0]
        if diagonal_result in (-3,3):
            self.finished=True
            self.result= 'win' if diagonal_result>0 else 'loss'
        
        #If we placed something on all places without a victor then it's a draw
        if 0 not in self.state and self.result is None:
            self.finished=True
            self.result='draw'
        
        
    def play(self,move=None):
        #row_index = int(input('Enter row 0-2: '))
        #column_index = int(input('Enter column 0-2: '))
        row_index = np.random.randint(0,3)
        column_index = np.random.randint(0,3)
        if move is None:
            move = (row_index,column_index)
        
        if row_index>2 or column_index>2:
            #print('out of bounds try again')
            return self.play()
        if self.state[move]!=0:
            #print('someone already played this position, try again')
            return self.play()
        previous_state = self.state
        self.state[move] = self.to_play
        self.move_taken = move
        self.game_status()
        self.to_play = self.to_play*-1
        if self.finished:
            return



#They have a nodeclass in Muzerro that looks like this
#Not really sure how that works
#So I assume each node has children that are other nodes.
#And each node then have attributes, like score, rewards, who's turn it is to play etc
#THat actually makes a lot of sense!!!!
#So I should be able to run an expansion function on this, to then update the children
#something like this

"""root = Node(0)
    current_observation = game.make_image(-1)
    expand_node(root, game.to_play(), game.legal_actions(),
                network.initial_inference(current_observation))
    add_exploration_noise(config, root)"""
#Note that root node is the first node of the entire treee that has no parents
#A leaf node is a node that doesn't have any children

"""
This process continues, creating a new MCTS tree from scratch each turn and using it to choose an action,
until the end of the game.

So I play an imaginary game from current board position each turn
during those imaginary games I run MCTS for X amount of simulations
It should then return some stuff, that allows me to pick the next turn.
After my next turn, and opponents next turn, I then re-run the MCTS from scratch, not keeping any old historic scores.

"""

class Node(object):

    def __init__(self, prior: float):
        self.visit_count = 0 #Amount of times we have simulated from this node or any of its children
        self.to_play = -1 #Whos turn it is to play, not sure why this isn't part of init parameters?
        self.prior = prior #the predicted prior probability of choosing the action that leads to this node
        self.value_sum = 0 #the backfilled value sum of the node 
        
        #This should be put by an expand_node function if I expand the node, each child being its own node
        #It should be a ditctionary containing the action taken as the key, and the child node that creates as value
        '''visit_counts = [
      (child.visit_count, action) for action, child in node.children.items()
      ]'''
        self.children = {} #{action:childnode}
        
        self.hidden_state = None # the hidden state the node corresponds to
        self.reward = 0 #the predicted reward received by moving to this node

    #This function just lets us know if we have expanded the node already
    def expanded(self) -> bool:
        return len(self.children) > 0

    #This seems to calculate the %ratio a visit of node resulted in win, which should be used for node selection criteria
    #Might also be used as reward for the ML algorithm?
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count



def expand_node(root : Node):
    #This is where we expand our current turn to understand which moves we can make
    moves = []
    current_state = root.hidden_state.copy()
    
    #this gives legal nodes
    #In actual muzero this is done elsewhere I think
    for r in range(0,3):
        for c in range(0,3):
            if current_state[r,c]==0:
                moves.append((r,c))
    #We add a childnode for each possible move
    for move in moves:
        new_state = current_state.copy()
        new_state[move] = root.to_play #The new state of the board if we play this move
        childnode = Node(0) #I create a new childnode, but not sure what to put as prior
        childnode.to_play = root.to_play*-1 #It's the next person's turn at this node
        childnode.hidden_state=new_state #The state of the childnode based on playing this move
        root.children.update({move:childnode}) #Add this node to our possible children nodes        
    #We return the new node now containing its expanded children
    return root



def choose_action(root:Node,legal_moves:list):
    #probably thinking a bit incorrectly here
    #I assume that just the first child selection should be UCB based
    #Next selections should be random?
    #Or maybe not necessairly
    action = None
    #If we havent started yet, then just take random move
    if root.visit_count==0:
        while action not in legal_moves:
            row_index = np.random.randint(0,3)
            column_index = np.random.randint(0,3)
            action = (row_index,column_index)
        return action
    #if we have started then calculate the ucb score
    #for each child node and select highest score
    exploration_values = []
    actions = [] #Will store the actions so we can get child
    for action_key in root.children:
        if root.children[action_key].visit_count==0:
            child_score = float('inf')
        else:
            child_score = ucb_score(parent=root,child=root.children[action_key])
        exploration_values.append(child_score)
        actions.append(action_key)
    highest_value = max(exploration_values)
    action = actions[exploration_values.index(highest_value)]
    return action


def add_exploration_noise(node: Node):
    # Root prior exploration noise.
    root_dirichlet_alpha = 1 #0.3 is for chess, not sure if shoul have lower or higher
    root_exploration_fraction = 0.25
    #add exploration noise at start of search
    actions = list(node.children.keys())
    noise = np.random.dirichlet([root_dirichlet_alpha] * len(actions))
    frac = root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

def ucb_score(parent: Node, child: Node) -> float:
#     pb_c_base = 19652
#     pb_c_init = 1.25
#     pb_c = math.log((parent.visit_count + pb_c_base + 1) /
#                   pb_c_base) + pb_c_init
#     pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
#     #print('pc_c',pb_c)
#     prior_score = pb_c * child.prior
#     #print('prior_score',prior_score)
#     value_score = child.value()
#     print('returned_value',prior_score + value_score)
#     return prior_score + value_score
    parent_node_visits = parent.visit_count

    exploration_term = (math.sqrt(2.0)
                        * math.sqrt(math.log(parent_node_visits) / child.visit_count))

    value = child.value() + exploration_term
    return value


def run_shit(root:Node,game:TicTacToe,simulations=1200):
    #This function runs bunch of simulations
    #It then return the move that returned most wins
    #We will try random move 100 times
    for _ in range(simulations):
        #This will play the next move and expand till the end
        fake_game = TicTacToe()
        fake_game.to_play = game.to_play
        fake_game.state = game.state.copy()
        path = [root] #This contains all the nodes we played
        legal_moves = [list(root.children.keys())][0]
        action = choose_action(root,legal_moves)
        current_node = root.children[action]
        while fake_game.result is None:
            path.append(current_node)
            fake_game.play(action)
            if fake_game.result is not None:
                break
            if not current_node.expanded():
                expand_node(current_node)
                add_exploration_noise(current_node)
            legal_moves = [list(current_node.children.keys())][0]
            action = choose_action(current_node,legal_moves)
            #The node corresonding to chosen action
            chosen_node = current_node.children[action]
            current_node = chosen_node

        # we give the result to parent node or all nodes?
        # probably just parent?    
        #back propagate result
        #print(path)
        for node in reversed(path):
            #print(node)
            #print(node.to_play)
            node.visit_count+=1
            if fake_game.result=='win' and node.to_play==-1:
                node.value_sum+=1
            #in tictactoe draw is pretty much expected so we give half a point to all
            if fake_game.result=='draw':
                node.value_sum+=0.5
            #If the other one loses then we get an extra point
            if fake_game.result=='loss' and node.to_play==1:
                node.value_sum+=1

    actions = []
    values = []
    result = {}
    visits = []
    nodes = []
    for action,child in root.children.items():
        nodes.append(child)
        actions.append(action)
        values.append(child.value())
        result.update({action:child.value()})
        visits.append(child.visit_count)
#     return actions[values.index(max(values))],result
    return actions[visits.index(max(visits))],result,visits,nodes

def play_game(turns=9,simulations=1200):
    game = TicTacToe()
    counter = 0
    History=[]
    while game.result is None and counter<turns:
        counter+=1
        if game.to_play==-1:
            root = Node(0) #We create a new root node from scratch
            root.to_play = game.to_play #We give it turn to play based on game's current status
            root.hidden_state = game.state.copy() #We give it the current state of the game
            expand_node(root) #Each time we start new search we expand root
            add_exploration_noise(root) #We then add exploration noise as prior value to children
            move,values,visits,child = run_shit(root,game,simulations) #Simulate some games
            game.play(move)
            History.append([move,values,visits,child,game.state.copy()])
        else:
            row_index = int(input('Enter row 0-2: '))
            column_index = int(input('Enter column 0-2: '))
            move = (row_index,column_index)
            game.play(move)
        print(game.state)
    print(game.result)
    return History

def self_play(number_of_games:int):
    Games = []
    for _ in range(number_of_games): #play some real games to see if it always ties
        History = []
        game = TicTacToe()
        while game.result is None:
            #Initialize a new round of MCTS
            root = Node(0) #We create a new root node from scratch
            root.to_play = game.to_play #We give it turn to play based on game's current status
            root.hidden_state = game.state.copy() #We give it the current state of the game
            expand_node(root) #Each time we start new search we expand root
            add_exploration_noise(root) #We then add exploration noise as prior value to children
            move,values,visits,child = run_shit(root,game) #Simulate some games

            game.play(move)
            History.append([move,values,visits,child,game.state.copy()])
        Games.append([game.result,History])
    return Games



if __name__ == '__main__':
    Info = play_game(simulations=1200,turns=9)
    
