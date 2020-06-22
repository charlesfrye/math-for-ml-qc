from client.api.notebook import Notebook
import wandb


class WandbTrackedOK(object):

    def __init__(self, entity, path, project):
        self.grader = Notebook(path)
        wandb.init(entity=entity, project=project, anonymous="must")
        self.test_map = self.grader.assignment.test_map
        self.pass_dict = {k: 0 for k in self.test_map}
        self.log()

    def grade(self, question, *args, **kwargs):
        result = self.grader.grade(question, *args, **kwargs)
        self.pass_dict[question] = result["passed"]
        self.log()

    def log(self):
        total = sum([v for v in self.pass_dict.values()])
        wandb.log({"passes": self.pass_dict,
                   "total": total})

    def __delete__(self):
        wandb.join()
