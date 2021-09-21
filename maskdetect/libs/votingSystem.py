from typing import MutableMapping
from utils.resultManager import maskToken_to_dangerLevel
from utils.types import MaskToken


class VotingStatus : 
    limitVote = 100 
    initVote = 2 
    conNFLimit = 5
    
    def __init__(self) : 
        self.votes = { 
            MaskToken.FaceNotFound : VotingStatus.initVote, 
            MaskToken.Masked : 0 , 
            MaskToken.NotMasked : 0
        } 
        self.continuedNF = 0 
        self.voteResult = MaskToken.FaceNotFound  
    
    def getVoteResult(self) -> MaskToken : 
        return self.voteResult  
        
    def vote(self, mtoken) -> MaskToken :
        
        # this is unusuall cases -> NotNear or UnKnown don't need a vote. 
        if mtoken == MaskToken.NotNear : 
            return self.voteResult 
        if mtoken == MaskToken.UnKnown : 
            return self.voteResult
        
        # faceNotFound -> +1 continuedNF 
        if mtoken == MaskToken.FaceNotFound : 
            self.continuedNF += 1; 
            if self.continuedNF >= VotingStatus.conNFLimit : 
                return MaskToken.FaceNotFound; 
            return self.voteResult 
        
        # others  
        voteWeight = 1
        if mtoken == MaskToken.NotMasked : 
            voteWeight = 5
        self.continuedNF = 0 
        if (self.votes[mtoken] + voteWeight) < VotingStatus.limitVote : 
            self.votes[mtoken] += voteWeight  
        else :  # if voting overflows the limit, divide by 2 to continue, and give opportunity for others,, 
            self.votes[MaskToken.Masked] = self.votes[MaskToken.Masked] / 2
            self.votes[MaskToken.NotMasked] = self.votes[MaskToken.NotMasked] / 2 
            self.votes[mtoken] += voteWeight  
            
        if self.voteResult != mtoken : 
            if  self.votes[self.voteResult] < self.votes[mtoken] : 
                self.voteResult = mtoken 
        return self.voteResult 
    
    def show(self) : 
        print("Masked       \t : {}".format(self.votes[MaskToken.Masked]))        
        print("NotMasked    \t : {}".format(self.votes[MaskToken.NotMasked]))        


class VotingSystem : 
    def __init__(self) : 
        self.vstatus = {}
    
    def getVote(self, tid) -> MaskToken : 
        if self.vstatus.get(tid) is None : 
            return MaskToken.FaceNotFound  #there is not vote before.... =<  
        
        result = self.vstatus[tid].getVoteResult() 
        return result 
        
        
    def vote(self, tid, mtoken) -> MaskToken : 
        if self.vstatus.get(tid) is None : 
            self.vstatus[tid] = VotingStatus() 
        
        result = self.vstatus[tid].vote(mtoken)
        return result 
        
    def show(self, tid) : 
        if self.vstatus.get(tid) is None : 
            print("no such ID_{} ".format(tid))
        else : 
            print("== TID_{} result ==".format(tid))
            self.vstatus[tid].show()
            print("===================")