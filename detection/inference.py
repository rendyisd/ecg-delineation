# DEPRECATED

class Rule:
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_EQUALS = ">="
    LESS_THAN_EQUALS = "<="

    def __init__(self, conditions, consequence):
        self.conditions = conditions
        self.consequence = consequence
        
    def evaluate(self, facts):
        for var, constraints in self.conditions.items():
            temp_bool = False
            for constraint in constraints:
                op, val = constraint[0], constraint[1]

                temp_bool = self._check_constraint(var, op, val, facts)
                if temp_bool is True:
                    break
            
            if temp_bool is False:
                return None
        
        return self.consequence


    def _check_constraint(self, var, op, val, facts):
        if op == self.EQUALS:
            return facts.get(var) == val
        elif op == self.NOT_EQUALS:
            return facts.get(var) != val
        elif op == self.GREATER_THAN:
            return facts.get(var) > val
        elif op == self.LESS_THAN:
            return facts.get(var) < val
        elif op == self.GREATER_THAN_EQUALS:
            return facts.get(var) >= val
        elif op == self.LESS_THAN_EQUALS:
            return facts.get(var) <= val
        return False

class InferenceSystem:
    def __init__(self):
        self.rules = []

    def add_rule(self, conditions, consequence):
        self.rules.append(Rule(conditions, consequence))

    def infer(self, facts):
        for rule in self.rules:
            result = rule.evaluate(facts)
            if result:
                return result
        return None