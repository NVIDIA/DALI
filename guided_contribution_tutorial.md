# Guided contribution tutorial

DALI is an open-source project, therefore we are more than happy to accept external contributions (refer to [CONTRIBUTION.md](https://github.com/NVIDIA/DALI/blob/main/CONTRIBUTING.md) for details about regular contributions). Moreover, we are also keen on closer and more structured cooperation. For instance, you can work on a more significant contribution to DALI as a part of your school project, Deep Learning oriented university class, Bachelor's or Master thesis, etc. We also recommend such project to be implemented by a group of developers, so if you have any friends who would like to join you, that's even better. This guide outlines the whole process, from scratch to your first merged PR.

1. If you are ready to commit more time to develop DALI, please contact us directly. You can use either Github issues or write to us on [via email](mailto:dali-team@nvidia.com). We will agree upon the contribution goal, make the plan, sign the necessary NDA (Non-disclosure agreement) and establish a more direct communication channel.
1. I'm assuming, that we have already set up some communication channel (e.g. Slack, mail, or other). If we didn't yet, please ask your contact point in our team to do so.
1. You have been assigned with a DALI-team developer, who would help you with everything you need. Let's call him The Mentor. That shall be your primary contact point with us. Nevertheless, feel free to reach out to anyone from the team, we all are happy to help.
1. Whenever you're in doubt, please reach out to The Mentor. We are here to help you and we are happy to help you. There are problems that we faced before and we know how to handle them. Reinventing the wheel is not as exciting as solving problems nobody solved before!
1. Discuss with our team, what would be the topic of your contribution. This part is really important and you shall work closely with The Mentor on the contribution idea. We have some internal documents and process that we would like to keep, and The Mentor is there to help you. Also, The Mentor would contact with any internal stakeholders, to make sure that the contribution is well designed.
1. You can apply for a workstation from our pool. Please contact The Mentor for any details and access rights. The workstation is equipped with a GPU that's suitable for development. It also might be used for some benchmarking and other tests. We encourage you to develop DALI there, still you can use your own machine if you like. And please, do not engage this workstation in some indecent acts, like crypto mining. We do monitor the workloads and we are able to spot the improper ones.
1. That's the moment we can roll up the sleeves and get to work. Your first step would be to fork DALI repository in the Github. Just go to our repository [main page](https://github.com/NVIDIA/DALI) and click "Fork" in the upper right corner. In DALI we follow the pattern, that we push the code to our forks and then create a PR in main DALI repository. Typically, you would like to clone your fork and then add an upstream remote to this clone, to pull and changes made to main DALI, something around these lines:

        git clone --recursive https://github.com/<your_name>/DALI
        cd DALI
        git remote add upstream https://github.com/NVIDIA/DALI

1. When the DALI repository is cloned, please follow the [bare-metal build guide](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/compilation.html#bare-metal-build). Some contribution topics would not require full bare-metal build, so sometimes the [docker build](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/compilation.html#using-docker-builder-recommended) could do. However, we encourage the bare-metal one, as it is way better for development, profiling, debugging, etc. Please carefully follow the tutorial and conduct every part (including "Verify the build").
1. Before starting coding, we encourage you to read the [contribution guide](https://github.com/NVIDIA/DALI/blob/main/CONTRIBUTING.md) and [code-style guide](https://github.com/NVIDIA/DALI/blob/main/STYLE_GUIDE.md) located in DALI repository. All the contributions must follow these guidelines.
1. Your typical git workflow should be as follows:

        # Create a branch with your feature
        git checkout -b my_awesome_feature

        # Write code, write tests, debug, develop…

        # From time to time you should get up to date with the main DALI branch:
        git checkout main
        git pull upstream main
        git checkout -b my_awesome_feature
        git rebase main

1. We strongly encourage to create a draft PR, although this step isn't strictly necessary. This way you can save your work progres, we can evaluate if the development direction makes sense and we can sync on the ongoing work and apply any necessary changes as-early-as-possible. You don't need to fill the PR template, description and other things for the draft. After creating the Draft PR, you may also ask The Mentor to trigger a CI run on your PR. It's a good practice to fix all the CI problems that can occur, before the code-review.
1. When your PR is ready for review, please make sure, that the PR template is properly filled and that the PR description fully explains your contribution (a good example can be found [here](https://github.com/NVIDIA/DALI/pull/3557)). If so, please mark the PR in Github as "Ready for review". During the review process, the PR will be assigned with reviewers. All assigned reviewers have to approve the PR before it is merged. Please work with reviewers on polishing the PR and applying the necessary changes.
1. You've got all the green ticks from the assigned reviewers? That great, congrats! That's a moment to reach out to The Mentor to ask for final CI run. When the CI passes The Mentor should merge your PR. Congratulations, you have contributed to DALI!

