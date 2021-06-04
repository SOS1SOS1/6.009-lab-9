"""6.009 Lab 10: Snek Interpreter Part 2"""

import sys
sys.setrecursionlimit(5000)

"""6.009 Lab 9: Snek Interpreter"""

import doctest
# NO ADDITIONAL IMPORTS!


###########################
# Environments #
###########################

class Environment():
    """
    An environment consists of bindings from variable names to values, and possibly 
    a parent environment, from which other bindings are inherited. One can look up a 
    name in an environment, and one can bind names to values in an environment.
    """
    def __init__(self, parent_env, bindings=None):
        self.bindings = { key:value for key, value in bindings.items() } if bindings else {}
        self.parent = parent_env

    def set_value(self, name, expr):
        # variable name cannot be a number or contain parentheses or spaces
        if is_number(name) or ")" in name or "(" in name or " " in name:
            raise SnekSyntaxError
        self.bindings[name] = expr

    def get_value(self, name):
        if name in self.bindings:
            # name has a binding in the environment
            return self.bindings[name]
        if self.parent:
            # name did not have a binding in this env, look in its parent
            return self.parent.get_value(name)
        # name does not have a binding in the env and no more parents
        raise SnekNameError

    def del_value(self, name):
        # if variable is not bound locally, then it raises an error
        if not name in self.bindings:
            raise SnekNameError("tried deleting variable that isn't bound here")
        value = self.bindings[name]
        del self.bindings[name]
        return value

    def update_value(self, name, new_val):
        if name in self.bindings:
            # name has a binding in the environment
            self.bindings[name] = new_val
            return new_val
        if self.parent:
            # name did not have a binding in this env, look in its parent
            return self.parent.update_value(name, new_val)
        # name does not have a binding in the env and no more parents
        raise SnekNameError


###########################
# Functions #
###########################

class Function:
    """
    Represent your user-defined functions. Stores the code representing the body of the 
    function, the names of the function's parameters, and a pointer to the environment 
    where the function was defined
    """
    def __init__(self, params, body, env):
        """
        params: array of parameters for function
        body: parsed expression
        env: environment where the function was defined
        """
        self.params = params
        self.body = body
        self.env = env


###########################
# Pair #
###########################

class Pair:
    """
    Represent your user-defined functions. Stores the code representing the body of the 
    function, the names of the function's parameters, and a pointer to the environment 
    where the function was defined
    """
    def __init__(self, car, cdr):
        """
        car: first element in the pair
        cdr: second element in the pair
        """
        self.car = car
        self.cdr = cdr


######################
# Built-in Functions #
######################

def product(args):
    """
    snek built-in for product of numbers
    """
    value = args[0]
    for l in args[1:]:
        value *= l
    return value

def divide(args):
    """
    snek built-in for division of numbers
    """
    value = args[0]
    for l in args[1:]:
        value /= l
    return value

### Snek built-ins for comparisons ###
def equal(cur_val, prev):
    return cur_val == prev

def decreasing(cur_val, prev):
    return cur_val < prev

def increasing(cur_val, prev):
    return cur_val > prev

def nondecreasing(cur_val, prev):
    return cur_val >= prev

def nonincreasing(cur_val, prev):
    return cur_val <= prev

def order(args, cond_func):
    def check_order(args):
        prev = args[0]
        for l in args[1:]:
            if not cond_func(l, prev):
                return snek_builtins["#f"]
            prev = l
        return snek_builtins["#t"]  
    return check_order(args)


def create_list(args):
    """
    Takes in list of values and returns a cons cell list representation of them
    """
    if len(args) == 0:
        # empty list
        return snek_builtins['nil']
    cons_list = Pair(args[0], snek_builtins['nil'])
    cur_pair = cons_list
    for v in args[1:]:
        cur_pair.cdr = Pair(v, snek_builtins['nil'])
        cur_pair = cur_pair.cdr
    return cons_list


def get_car(pair):
    """
    If the input is a pair, then it returns the car of the cons cell
    """
    if not type(pair) == Pair:
        raise SnekEvaluationError("tried to get car on something that isn't a cons cell")
    return pair.car

def get_cdr(pair):
    """
    If the input is a pair, then it returns the cdr of the cons cell
    """
    if not type(pair) == Pair:
        raise SnekEvaluationError("tried to get cdr on something that isn't a cons cell")
    return pair.cdr

def get_length(l):
    """
    Returns the length of the cons cell list
    """
    # empty list
    if not l:
        return 0
    # l is not a list
    if type(l) != Pair or (l.cdr and type(l.cdr) != Pair):
        raise SnekEvaluationError("tried to get length on something that isn't a list")
    cur = l
    length = 0
    while type(cur) == Pair:
        cur = cur.cdr
        length += 1
    return length

def get_elt_at(l, index):
    """
    Returns the element at the specified index in the list l
    """
    # l is empty or not a list or index is negative
    if not l or type(l) != Pair or index < 0:
        raise SnekEvaluationError("tried to get element at index of something is not a cons cell or list")
    cur = l
    cur_i = 0
    while type(cur) == Pair:
        if cur_i == index:
            return cur.car
        cur = cur.cdr
        cur_i += 1
    raise SnekEvaluationError("index out of bounds")

def get_list_values(l):
    """
    Yields the values in a cons cell list
    """
    if not l:
        # empty list
        return snek_builtins['nil']
    # l is not a list
    if type(l) != Pair or (l.cdr and type(l.cdr) != Pair):
        raise SnekEvaluationError("tried to get length on something that isn't a list")
    yield l.car
    cur = l
    while type(cur) == Pair:
        if type(cur.cdr) == Pair:
            yield cur.cdr.car
        cur = cur.cdr

def concat(lists):
    """
    Concatenates multiple lists together as a new list and returns that new list
    """
    if len(lists) == 0:
        # empty list
        return snek_builtins['nil']
    new_list, cur_pair = None, None
    for l in lists:
        if l != None:
            for val in get_list_values(l):
                if not new_list:
                    new_list = Pair(val, snek_builtins['nil'])
                    cur_pair = new_list
                else:
                    cur_pair.cdr = Pair(val, snek_builtins['nil'])
                    cur_pair = cur_pair.cdr
    return new_list

def snek_map(func, l):
    """
    Takes a function and a list as arguments, and it returns a new list containing 
    the results of applying the given function to each element of the given list
    """
    new_list, cur_pair = None, None
    for val in get_list_values(l):
        if type(func) == Function:
            mapped_val = call_function(func, [val], built_ins_env)
        else:
            mapped_val = call_built_in_function(func, [val], built_ins_env)
        if not new_list:
            new_list = Pair(mapped_val, snek_builtins['nil'])
            cur_pair = new_list
        else:
            cur_pair.cdr = Pair(mapped_val, snek_builtins['nil'])
            cur_pair = cur_pair.cdr
    return new_list

def snek_filter(func, l):
    """
    Takes a function and a list as arguments, and it returns a new list containing 
    only the elements of the given list for which the given function returns true
    """
    new_list, cur_pair = None, None
    for val in get_list_values(l):
        if type(func) == Function:
            result = call_function(func, [val], built_ins_env)
        else:
            result = call_built_in_function(func, [val], built_ins_env)
        if result:
            if not new_list:
                new_list = Pair(val, snek_builtins['nil'])
                cur_pair = new_list
            else:
                cur_pair.cdr = Pair(val, snek_builtins['nil'])
                cur_pair = cur_pair.cdr
    return new_list

def snek_reduce(func, l, initial_val):
    """
    Takes a function, a list, and an initial value as inputs. It produces its output by 
    successively applying the given function to the elements in the list, maintaining an 
    intermediate result along the way
    """
    result_so_far = initial_val
    for val in get_list_values(l):
        if type(func) == Function:
            result_so_far = call_function(func, [result_so_far, val], built_ins_env)
        else:
            result_so_far = call_built_in_function(func, [result_so_far, val], built_ins_env)
    return result_so_far

snek_builtins = {
    '#t': True,
    '#f': False,
    'nil': None,
    '+': sum,
    '-': lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    '*': product,
    '/': divide,
    '=?': lambda args: order(args, equal),
    '>': lambda args: order(args, decreasing),
    '>=': lambda args: order(args, nonincreasing),
    '<': lambda args: order(args, increasing),
    '<=': lambda args: order(args, nondecreasing),
    'not': lambda arg: snek_builtins["#f"] if arg[0] else snek_builtins["#t"],
    'cons': lambda args: Pair(*args),
    'list': lambda args: create_list(args),
    'car': lambda args: get_car(args[0]),
    'cdr': lambda args: get_cdr(args[0]),
    'length': lambda args: get_length(args[0]),
    'elt-at-index': lambda args: get_elt_at(*args),
    'concat': lambda args: concat(args),
    'map': lambda args: snek_map(*args),
    'filter': lambda args: snek_filter(*args),
    'reduce': lambda args: snek_reduce(*args),
    'begin': lambda args: args[-1]
}
built_ins_env = Environment(None, snek_builtins)


###########################
# Snek-related Exceptions #
###########################

class SnekError(Exception):
    """
    A type of exception to be raised if there is an error with a Snek
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """
    pass


class SnekSyntaxError(SnekError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """
    pass


class SnekNameError(SnekError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """
    pass


class SnekEvaluationError(SnekError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SnekNameError.
    """
    pass


############################
# Tokenization and Parsing #
############################

def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Snek
                      expression
    """
    tokens = []
    last_char = ""
    comment_start_i = -1
    for idx, c in enumerate(source):
        if comment_start_i > 0:
            if c == "\n":
                # remove all text in comment
                tokens = tokens[:comment_start_i]
                comment_start_i = -1
        else:
            if c in {"(", ")"}:
                tokens.append(c)
            elif c == ";":
                # start of comment
                comment_start_i = len(tokens)
            elif c != " " and c != "\n":
                if tokens and last_char != "(" and last_char != " "  and last_char != "\n":
                    tokens[-1] = tokens[-1] + c
                else:
                    tokens.append(c)
        last_char = c
    return tokens

def is_number(token):
    """
    Returns true if the inputed token is a number, false otherwise
    """
    try:
        float(token)
    except ValueError:
        return False
    return True

def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    def parse_expression(index):
        """
        Takes in an index into the tokens list and returns the expression found 
        starting at the location given by index (an instance of one of the Symbol subclasses)
        and the index beyond where this expression ends
        """
        token = tokens[index]
        if is_number(token):
            # token represents an integer
            if "." in token:
                return float(token), index+1
            else:
                return int(token), index+1
        elif token == '(':
            end_index = index + 1
            sub_exprs = []
            var_definition = False
            func_definition = False
            while end_index < len(tokens) and tokens[end_index] != ")":
                s, end_index = parse_expression(end_index)
                sub_exprs.append(s)
                if s == ":=":
                    var_definition = True
                elif s == "function":
                    func_definition = True
            # check that variable definition is of correct format (:= NAME EXPR)
            if var_definition:
                if len(sub_exprs) != 3:
                    raise SnekSyntaxError
                    # type(sub_exprs[1]) != str and (
                if type(sub_exprs[1]) == list:
                    # if it is a list, then it must be a list of strings
                    if len(sub_exprs[1]) == 0:
                        raise SnekSyntaxError
                    for p in sub_exprs[1]:
                        if type(p) != str:
                            raise SnekSyntaxError
                elif type(sub_exprs[1]) != str:
                    raise SnekSyntaxError()
            # check that function definition is of correct format (:= NAME EXPR)
            if func_definition:
                # all functino definitions should have three elements and the params should be an array
                if len(sub_exprs) != 3 or type(sub_exprs[1]) != list:
                    raise SnekSyntaxError
                # check its a valid list of params (0 or more strings representing parameter names)
                for p in sub_exprs[1]:
                    if type(p) != str:
                        raise SnekSyntaxError
            # mismatched paratheses
            if end_index == len(tokens):
                raise SnekSyntaxError
            return sub_exprs, end_index+1
        elif token == ')':
            # mismatched paratheses
            raise SnekSyntaxError
        else:
            # token represents a string
            return token, index+1
    parsed_expression, next_index = parse_expression(0)
    if parsed_expression == ":=" or next_index < len(tokens):
        raise SnekSyntaxError
    return parsed_expression


##############
# Evaluation #
##############


### Evaluates Helper Functions ###

def define_variables(tree, env):
    name = tree[1]
    value = None
    # if the NAME in a := expression is itself an S-expression, it is implicitly 
    # translated to a function definition before binding
    if type(name) == list:
        # defining function
        name = tree[1][0]
        params = tree[1][1:]
        body = tree[2]
        value = Function(params, body, env)
    else:
        # defining variable
        value = evaluate(tree[2], env)
    env.set_value(name, value)
    return value

def define_function(tree, env):
    params = tree[1]
    body = tree[2]
    return Function(params, body, env)

def evaluate_if_statement(tree, env):
    # start of an if expression which has the special form of (if COND TRUEEXP FALSEEXP)
    cond = tree[1]
    result = evaluate(cond, env)
    if result:
        true_exp = tree[2]
        return evaluate(true_exp, env)
        # return true_exp
    false_exp = tree[3]
    return evaluate(false_exp, env)
    # return false_exp

def evaluate_func(tree, env):
    # calling built-in function or user-defined function
    vals = []
    value = None
    for e in tree[1:]:
        vals.append(evaluate(e, env))
    func = evaluate(tree[0], env)
    if type(func) == Function:
        value = call_function(func, vals, env)
    else:
        # calling built-in function
        value = call_built_in_function(func, vals, env)
    return value


def call_built_in_function(func, params, env):
    return func(params)

def call_function(func, params, env):
    # calling user-defined function
    if len(params) != len(func.params):
        raise SnekEvaluationError
    # make a new environment whose parent is the environment in which the function was defined
    new_env = Environment(func.env)
    # bind the function's parameters to the arguments that are passed to it
    for p, arg in zip(func.params, params):
        new_env.set_value(p, arg)
    # evaluate the body of the function in that new environment
    return evaluate(func.body, new_env)

# short-circuiting, if anything in and is false, then the entire thing is false
def evaluate_and_bool(tree, env):
    for arg in tree[1:]:
        if not evaluate(arg, env):
            return snek_builtins["#f"]
    return snek_builtins["#t"]

# short-circuiting, if anything in or is true, then the entire thing is true
def evaluate_or_bool(tree, env):
    for arg in tree[1:]:
        if evaluate(arg, env):
            return snek_builtins["#t"]
    return snek_builtins["#f"]

def evaluate_del(tree, env):
    """
    Used for deleting variable bindings within the current environment
    """
    return env.del_value(tree[1])

def evaluate_let(tree, env):
    """
    Used for creating local variable definitions
    """
    new_env = Environment(env)
    for var in tree[1]:
        val = evaluate(var[1], env)
        new_env.set_value(var[0], val)
    return evaluate(tree[2], new_env)


def evaluate_set(tree, env):
    """
    Pronounced "set bang", is used for changing the value of an existing variable
    """
    var = tree[1]
    new_val = evaluate(tree[2], env)
    env.update_value(var, new_val)
    return new_val

def evaluate(tree, env=None):
    """
    Evaluate the given syntax tree according to the rules of the Snek
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    # S-expression
    if type(tree) == list:
        # if the first element is not a valid function, raise an error
        special_forms = {":=", "function", "if", "and", "or", "del", "let", "set!"}
        if len(tree) == 0 or (type(tree[0]) != list and not tree[0] in special_forms and not tree[0] in snek_builtins and not evaluate(tree[0], env)) or type(tree[0]) == int:
            raise SnekEvaluationError("error while evaluating")
        if tree[0] == ":=":
            return define_variables(tree, env)
        elif tree[0] == "function":
            return define_function(tree, env)
        elif tree[0] == "if":
            return evaluate_if_statement(tree, env)
        elif tree[0] == "and":
            return evaluate_and_bool(tree, env)
        elif tree[0] == "or":
            return evaluate_or_bool(tree, env)
        elif tree[0] == "del":
            return evaluate_del(tree, env)
        elif tree[0] == "let":
            return evaluate_let(tree, env)
        elif tree[0] == "set!":
            return evaluate_set(tree, env)
        else:
            return evaluate_func(tree, env)
    elif type(tree) == Pair:
        return tree
    elif is_number(tree):
        # expression is a number
        return tree
    elif env:
        # expression is a symbol
        return env.get_value(tree)
    elif tree in snek_builtins:
        # expression is a symbol representing a name in snek_builtins
        return snek_builtins[tree]

import os
def evaluate_file(file_name, env=None):
    """
    Takes in the name of the file to be evaluated and optionally the environment 
    in which to evaluate the expression
    Returns the result of evaluating the expression contained in the file
    """
    with open (file_name, 'r') as f:
        tokens = tokenize(f.read())
        parsed = parse(tokens)
        result, _ = result_and_env(parsed, env)
    return result


def result_and_env(tree, env=None):
    """
    Returns a tuple with two elements: the result of the evaluation and the environment 
    in which the expression was evaluated
    """
    if not env:
        env = Environment(built_ins_env)
    result = evaluate(tree, env)
    return (result, env)

def start_repl(env):
    """
    REPL (a "Read, Evaluate, Print Loop")
    Continually prompts the user for input until they type QUIT
    Until then, it:
        - accepts input from the user,
        - tokenizes and parses it,
        - evaluates it, and
        - prints the result.
    """
    user_input = ""
    while True:
        user_input = input("👻 >>> ")
        if user_input == "QUIT":
            break
        try:
            tokens = tokenize(user_input)
            parsed = parse(tokens)
            result = evaluate(parsed, env)
            print(result)
        except SnekSyntaxError as e:
            print(e)
        except SnekNameError as e:
            print(e)
        except SnekEvaluationError as e:
            print(e)

if __name__ == '__main__':

    global_env = Environment(built_ins_env)

    # python3 lab.py test_files/definitions.snek
    
    # check if any files need to be evaluated first
    for arg in sys.argv[1:]:
        evaluate_file(arg, global_env)

    start_repl(global_env)

    pass


# (begin
#     (:= x 7)
#     (:= y 9)
#     (:= (square x) (* x x))
#     (factorial (square 2))
# )

# tokenizer - splits it into the important pieces, getting rid of comments and new lines
# parser - goes through the tokens, checks for syntax errors and creates a representation of symbols, numbers, and s-expressions
# evaluate - for each s-expression a new call to evaluate will be made, which will 
    # bind x to 7, calls define_variables helper
    # bind y to 9, calls define_variables helper
    # define square function, calls define_function helper
    # calls factorial, creates a new environment #1
        # evaluates the parameters (square 2)
            # calls square, creates a new environment #2
                # bind the function's parameters to the arguments that are passed to it, bind x to 2
                # evaluates body in this new enviroment #2, returns 4
        # bind n to 4
        # evaluates body in new environment #1, returns 24

# main env
    # x is bound to 7
    # y is bound to 9
    # square is defined
    # factorial is defined

# new env #1
    # n is bound to 4

# new env #2
    # x is bound to 2