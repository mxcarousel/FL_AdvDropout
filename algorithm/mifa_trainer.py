from . import trainer_base
from tqdm import tqdm

class mifa_trainer(trainer_base):
    def train(self):
        # self.seed_init()
        for round in tqdm(range(self.global_round)):
            self.lr_update(round)
            # form \cal{S} tilde
            self.sample()
            # multi-cast
            for idx in self.active_client:
                self.X[idx].local(self.server.retrieve(),self.lr_latest)
            
            if self.observation:
                self.observe(round)
            else:
                self.dropout()
            
            self.server.mifa_agg(self.active_client,self.X,round,self.lr_latest)

            if round % self.print_every == 0 :
                self.server.evaluate(round)
        
        if self.observation:
            self.observe_save()
            
        self.summary_save()