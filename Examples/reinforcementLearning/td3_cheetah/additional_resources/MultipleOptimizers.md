# Multiple Optimizers with Perforated AI

Reinforcement Learning tasks are quite varied in structure, and require different networks to either have different optimizers, since they are
updated independently of one another, or they require different learning rates between the networks. However, PerforatedAI manages global state,
and does not support individual models getting dendrites separately. As a result, a natural question arises: How do I
get the benefits of dendritic optimization when I have multiple optimizers? The arguments made throughout this file naturally extend to `n` optimizers, but for the sake of simplicity, we just assume we have 2 networks throughout. 

We recommend to use the same type of optimizer if doing experiments that require multiple networks to be updated independently. We will briefly touch on the other cases, but will not provide the same coverage as for the same optimizer case.

## Case 1: Same Optimizer Type (Recommended)

If the 2 networks we want to add dendrites to use the same type of optimizer, then we have to make very minor adjustments to add dendrites to both networks and optimize them both! We follow the steps outlined in customization.md, but with some key differences:

	1. We explicitly provide the parameter groups to the optimizer
	2. We set both network A and B to use the same optimizer
	3. We add helper functions to disable gradient flow for network A when updating B and vice versa
	
These changes allow for dendrites to be added to both networks and have their learning rates be managed by PAI instead of externally!
	
    # Define our Pair Module
	class Pair(nn.Module):
	    def __init__(self, A, B):
	        super(Pair, self).__init__()
	        self.A = A
	        self.B = B
	
	# Initialize the networks
    modelA = create_modelA()
    modelB = create_modelB()
    model  = Pair(modelA, modelB)
    model  = UPA.initialize_pai(model)

    # Set the networks directly
    modelA = model.netA
    modelB = model.netB

    # Initialize the optimizer + scheduler
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

    # (NEW): We make the parameter groups for the optimizer explicitly
    optimArgs = {
        "params": [
            {"params": list(model.A.parameters()), "lr": lr_A},
            {"params": list(model.B.parameters()), "lr": lr_B}           
        ]
    }
    schedArgs = {'mode':'max', 'patience': 5}

    # Both networks share an optimizer
    optimizer, _    = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    self.A_opt      = optimizer
    self.B_opt      = optimizer

    # (NEW): Define helper functions to toggle gradient flow for A and B
    def set_A_training(self, mode):
        for param in self.model.A.parameters():
            param.requires_grad = mode

    def set_B_training(self, mode):
        for param in self.model.B.parameters():
            param.requires_grad = mode
    
    ...

    # (NEW): Toggle gradient for A when updating B and reset A's status when done
    self.set_A_training(False)

    self.B_opt.zero_grad()
    B_loss.backward()
    self.B_opt.step()

    self.set_A_training(True)

    ...

    # (NEW): Toggle gradient for B when updating A and reset B's status when done
    self.set_B_training(False)

    self.A_opt.zero_grad()
    A_loss.backward()
    self.A_opt.step()

    self.set_B_training(True)

## Case 2: Different Optimizers, Same Schedulers

This is a similar case as above, but we need to manage more state ourselves. Without loss of generality, let's assume network A uses Adam and network B uses SGD. The key difference is that when the optimizer is restructured, the scheduler has been stepped and our lr for network A should be different from that of network B, so we manually adjust it. Our code would look like this:

    # Define our Pair Module
    class Pair(nn.Module):
        def __init__(self, A, B):
            super(Pair, self).__init__()
            self.A = A
            self.B = B
    
    # Initialize the networks
    modelA = create_modelA()
    modelB = create_modelB()
    model  = Pair(modelA, modelB)
    model  = UPA.initialize_pai(model)

    # Set the networks directly
    modelA = model.netA
    modelB = model.netB

    # Initialize the optimizer + scheduler
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

    # (NEW): We only provide A's parameters to the optimizer
    optimArgs = {
        "params": [
            {"params": list(model.A.parameters()), "lr": lr_A},
        ]
    }
    schedArgs = {'mode':'max', 'patience': 5}

    # Both networks share an optimizer
    optimizer, _    = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    self.A_opt      = optimizer
    self.B_opt      = optim.SGD(model.B.parameters(), lr=lr_B)

    # (NOTE): We no longer need the helper functions for toggling gradient flow    
    
    ...

    # (NEW): Update the lr of B_opt whenever restructuring occurs

    elif restructured:
        print("Model was restructured by Perforated AI")

        # Reset optimizer with the same parameters used initially
        optimArgs = {
            "params": [
                {"params": list(pai_model.critic.parameters()), "lr": model.lr_A},
        }
        schedArgs = {'mode':'max', 'patience': 5}

        optimizer, _    = GPA.pai_tracker.setup_optimizer(pai_model, optimArgs, schedArgs)
        agent.actor_opt  = optimizer
        agent.critic_opt = optim.SGD(model.B.parameters(), lr = optimizer.param_groups[0]['lr'])

## Case 3: Different Optimizers, Different Schedulers (Not recommended)

The above 2 cases share the scheduler for the learning rate, which is important for dendritic optimization. When we want to schedule our learning rates separately for both networks, we're effectively scheduling the learning rate on our own terms, independent of what PerforatedAI is doing in the background to manage the dendrites and their learning rates. We do not recommend doing this, but it is possible by duplicating Case 2's code and simply managing the scheduler for Network B as desired.

## Additional Notes and Thoughts

We provide some notes and a train of thought showing how we came across this question and answer pair.

1. Why don't you add multiple optimizer/scheduler support?
    The PAI code base assumes a single optimizer/scheduler. There are actually type hints in the tracker file that imply
    support for multiple optimizers and schedulers, but it doesn't quite work out that way. PAI manages much of its
    state globally, and as a result, even if we had multiple optimizers and schedulers attached to the pai_tracker object,
    they would reduce to all being the same optimizer and scheduler. This is because the schedulers would update
    based on the same logic and on the same global state. Theoretically, the code base could be rewritten
    to support multiple optimizers/schedulers natively, but it would be quite inefficient given how simple it is to manage
    the optimizer/scheduler state from the user side.

2. Why don't you just duplicate the tracker logic in a separate class?
    We spent a long time attempting this, but it's actually redundant. The global state that PAI manages is a mix of
    open source and closed source functions and statistics. These functions are critical, as they relate to the dendrites and
    the optimization of them. This means that you cannot fully rewrite the tracker to be local. One might simply make the
    compromise of calling the global state functions where necessary to make a local rewrite possible. This seems reasonable,
    but there is a catch. Assume we have 2 networks for an RL task, an Actor and a Critic, and we want to maximize reward. 
    We can setup the optimizer to have separate LRs for each network, so the only difference would be the Scheduler. However, the Scheduler specifically would rely on shared global statistics that both the Actor and Critic would use to step their own Schedulers. This means that by using the same global statistics for the Actor and Critic, the schedulers would step at the same time and by the same factor because they use the same statistics to determine how to update the LR! As a result, there is no real point in managing 2 separate schedulers, and it would be easier to just leverage the PAI scheduler!
