class cls_POD:
    """ This class stores a 2D decomposition """
    def __init__(self,n1,n2,rank):
        self.n1=n1
        self.n2=n2
        self.rank=rank
        self._Sigma=np.zeros(rank)
        self._U=np.zeros(n1,rank)
        self._Phi=np.zeros(n2,rank)

    def setSigma(self,Sigma):
        if Sigma.size != self.rank:
            raise AttributeError("rank is incoherent"+str(Sigma.size)+"!="+str(self.rank))
        self._Sigma=Sigma

    def setU(self,U):
        if U.shape[0] != n1:
            raise AttributeError("n1 incoherent"+str(U.shape[0])+"!="+str(n1))
        if U.shape[1] != self.rank:
            raise AttributeError("rank is incoherent"+str(U.shape[1])+"!="+str(self.rank))
        self._U=U

    def setPhi(self,Phi):
        if Phi.shape[0] != n2:
            raise AttributeError("n1 incoherent"+str(Phi.shape[0])+"!="+str(n2))
        if Phi.shape[1] != self.rank:
            raise AttributeError("rank is incoherent"+str(Phi.shape[1])+"!="+str(self.rank))
        self._Phi=Phi

    def build_POD_approx(self,r=0):
        """
            This function computes the sum of the POD modes up to a given r
                $$ A = sum_i=1^r (sigma_i . X_i \times T_i) $$

                X       is a 2 way np.array that contains at least the r first POD spatial modes
                T       is a 2 way np.array that contains at least the r first POD temporal modes
                sigma   is a 1 way array that contains the weight term for each association
                r       is the number of modes taken into account (into the sum)

                retrurn a 2D array of dimension 2  : (shape(X[:,0]), shape(T[:,0])
        """
        n =  np.size(self._sigma)
        if (r == 0) : r = n
        #Both algorithm give the same result but the first one is way faster
        S = np.zeros(np.shape(self.U))
        S[:n,:n] = np.diag(self._sigma)
        Phi_T=np.transpose(self._Phi)
        A = np.dot(self._U[:,:],np.dot(S[:n,:r], Phi_T[:r,:]))

        return A

    def __repr__(self):
        repr="Canonical Tensor represenation"
        repr+="\n ------------------------------"
        repr+="\n N ="+str(self.N)
        repr+="\n nt="+str(self.nt)
        repr+="\n Space dim ="+str(self.dSpace)
        repr+="\n Rank ="+str(self._rank)
        repr+="\n ------------------------------"
        repr+="\n Singular values \n"+str(self.Phi)
        repr+="\n ------------------------------"
        repr+="\n Axis 1 modes \n"
        repr+=str(self.U)
        repr+="\n ------------------------------"
        repr+="\n Axis 2 modes \n"
        repr+=str(self.Phi)
        repr+="\n ------------------------------"
        repr+="\n n1 ="+str(self.n1)
        repr+="\n n2 ="+str(self.n2)
        repr+="\n Rank ="+str(self.rank)
        repr+="\n ------------------------------"
        repr+="\n----------End of canonical tensor----------"
        return repr
