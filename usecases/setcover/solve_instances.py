"""
    Utility methods to solve and evaluate the Set Cover instances.
"""

from gurobipy import Model, GRB, MVar, Env
import numpy as np
from gurobipy import Model, GRB

########################################################################################################################


def compute_cost(instance, decision_vars, not_satisfied_demands):
    """
    Compute the true cost of a solution for the MSC.
    :param instance: usecases.setcover.generate_instances.MinSetCover; the problem instance.
    :param decision_vars: numpy.array of shape (num_sets, ); the solution.
    :param not_satisfied_demands: numpy.array of shape (num_prods, ); product demands that were not satisfied.
    :return: float; the true cost.
    """
    # Compute cost for using the sets
    real_cost = instance.set_costs @ decision_vars

    # Compute the cost for not satisfied demands
    not_satisfied_demands = np.clip(not_satisfied_demands, a_min=0, a_max=None)
    not_satisfied_demands_cost = not_satisfied_demands @ instance.prod_costs

    cost = real_cost + not_satisfied_demands_cost

    return cost


########################################################################################################################

class MinSetCoverProblem:
    """
    Class for the Minimum Set Cover problem.
    """

    def __init__(self, instance, output_flag=0):
        # Create the model
        env = Env(empty=True)
        env.setParam("OutputFlag", output_flag)
        env.start()
        self._problem = Model(env=env)

        self._problem.setParam('OutputFlag', output_flag)

        # Add an integer variable for each set
        self._decision_vars = self._problem.addMVar(shape=instance.num_sets,
                                                    vtype=GRB.INTEGER,
                                                    lb=0,
                                                    name='Decision variables')

        # Demand satisfaction constraints
        self._problem.addConstr(instance.availability @ self._decision_vars >= instance.demands)

        # Set the objective function
        self._problem.setObjective(instance.set_costs @ self._decision_vars, GRB.MINIMIZE)

    def solve(self):
        """
        Solve the optimization problem.
        :return: numpy.array, float; solution and its objective value.
        """
        self._problem.optimize()
        status = self._problem.status

        assert status == GRB.Status.OPTIMAL, "Solution is not optimal"

        solution = self._decision_vars.X
        obj_val = self._problem.objVal

        print_str = ""
        for idx in range(len(solution)):
            print_str += f'Set n.{idx}: {solution[idx]} - '
        print_str += f'\nSolution cost: {obj_val}'

        return solution, obj_val
