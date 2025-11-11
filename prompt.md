I want you to build this codebase for running machine learning experiments. Specifically, I want to be able to train an ensemble of reward models of the form R(s,a,s') from preferences over pairs of trajectories. I want an ensemble because I have an idea for a novel transition epistemic uncertainty measure U(s,a,s') that I want to compare with existing methods, and this uses an ensemble. 

Firstly, I want you to write the first part of the codebase which:
 - Allows us to create any Markov Decision Process (starting with a super simple gridworld environment MDP).
 - Allows us to sample many pairs of trajectories from this MDP using a policy mapping states to distributions over actions (which we can define in a way that is super general).
 - Labels the trajectory pairs with ground truth preferences (based on the return of the trajectories).
 - Allows us to trian an ensemble of reward models using pytorch, each of the form R(s,a,s') which return a real number for each transition. 

Do not run any git commands, I want to check them before pushing changes.