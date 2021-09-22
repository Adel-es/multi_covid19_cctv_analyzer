class VotingSystem : 
    voteListSize = 100
    def __init__(self) :
        self.visited = []
        self.voteDict = {} 
        self.voteList = [] 
        self.count = 0
        self.voteResult = None
        
        for i in range(0, 100) : 
            self.visited.append(False)
    
    def _vote_decrease(self, tid) : 
        self.voteDict[tid] = self.voteDict[tid] - 1
        sortedDict = sorted(self.voteDict.items(), key=lambda x : x[1])
        if len(sortedDict) == 0 : 
            self.voteResult = None 
        else : 
            self.voteResult = sortedDict[0][1]


    def _vote_increase(self, tid) : 
        if self.voteDict.get(tid) is None : 
            self.voteDict[tid] = 1
        else : 
            self.voteDict[tid] = self.voteDict[tid] + 1
        
        if self.voteResult == None :
            self.voteResult = tid   
        elif self.voteDict[self.voteResult] < self.voteDict[tid] : 
                self.voteResult = tid 
        return self.voteResult 
    
    
    def vote(self, tid) -> int: 
        index = self.count % self.voteListSize; 
        self.count = self.count + 1 
        
        if self.visited[index] == True : 
            old_vote = self.voteList[index] 
            self._vote_decrease(old_vote)
        
        self.visited[index] = True 
        self.voteList[index] = tid
        return self._vote_increase(tid) 
