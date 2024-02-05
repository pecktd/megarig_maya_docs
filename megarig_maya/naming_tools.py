from typing import Union, Tuple

"""
A set of functions to manipulate a name of maya node
regarding the Naming Convention.

Naming Convention:
[side_]name_[desc1_][desc2_][desc..._]type

Example:
l_fk_arm_1_jnt
l_ik_arm_ctrl
chest_ctrl
l_ik_clav_ctrl
l_main_arm_ctrl
"""
SIDES = ("l", "r")


def tokenize(name: str) -> Tuple[str, list, str]:
    tokens = name.split("_")

    if len(tokens) < 2:
        raise ValueError("The given name {} is invalid.".format(name))

    if len(tokens) == 2:
        # Only name and type are in the given name.
        return "", tokens[0], tokens[1]
    else:
        if tokens[0] in SIDES:
            # The first token is a side.
            return tokens[0], tokens[1:-1], tokens[-1]
        else:
            # There is no side in the name.
            return "", tokens[:-1], tokens[-1]


def compose(side: str, names: Tuple[str], type_: str) -> str:
    name = "_".join([n for n in names if n])
    if side:
        return "{}_{}_{}".format(side, name, type_)
    else:
        return "{}_{}".format(name, type_)


def add_desc(name: str, desc: str, type_: Union[str, None]) -> str:
    side, names, otype = tokenize(name)
    names.append(desc)

    if type_:
        # If the new type is given.
        return compose(side, names, type_)
    else:
        return compose(side, names, otype)
