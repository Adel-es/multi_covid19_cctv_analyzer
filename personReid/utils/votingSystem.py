import logging

class VotingSystem : 
    voteListSize = 100
    def __init__(self) :
        self.visited = []
        self.voteDict = {} 
        self.voteList = [] 
        self.count = 0
        self.voteResult = None
        self.logger = logging.getLogger('root')
        
        for i in range(0, 100) : 
            self.voteList.append(-1)
            self.visited.append(False)
    
    def _vote_decrease(self, tid) : 
        self.voteDict[tid] = self.voteDict[tid] - 1
        sortedDict = sorted(self.voteDict.items(), key=lambda x : x[1], reverse=True)
        if len(sortedDict) == 0 : 
            self.voteResult = None 
        else : 
            self.voteResult = sortedDict[0][0]


    def _vote_increase(self, tid) : 
        if self.voteDict.get(tid) is None : 
            self.voteDict[tid] = 1
        else : 
            self.voteDict[tid] = self.voteDict[tid] + 1

        if self.voteResult == None :
            self.voteResult = tid   
        elif self.voteDict[self.voteResult] < self.voteDict[tid] : 
                self.voteResult = tid 
        logging.debug("Reid Vote State : {}".format(self.voteDict))
        return self.voteResult 
    
    
    def vote(self, tid) : 
        index = self.count % self.voteListSize; 
        self.count = self.count + 1 
        
        if self.visited[index] == True : 
            old_vote = self.voteList[index] 
            self._vote_decrease(old_vote)
        
        self.visited[index] = True 
        self.voteList[index] = tid
        return self._vote_increase(tid) 
