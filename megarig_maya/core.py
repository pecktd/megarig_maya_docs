# Python modules.
import re
from typing import Union, List, Tuple
import dataclasses

# Maya modules.
import maya.cmds as mc
import maya.OpenMaya as om

# megarig modules.
from megarig_maya import naming_tools
from megarig_maya import curve_param


SKIN_SET = "skin_set"
FACIAL_SKIN_SET = "faial_skin_set"


@dataclasses.dataclass
class Operator:
    """Operators on utility nodes."""

    # plusMinusAverage
    no_operation = 0
    sum = 1
    subtract = 2
    average = 3

    # multiplyDivide
    multiply = 1
    divide = 2
    power = 3

    # condition
    equal = 0
    not_equal = 1
    greater_than = 2
    greater_or_equal = 3
    less_than = 4
    less_or_equal = 5

    # vectorProduct
    no = 0
    dot = 1
    cross = 2
    vector_mat = 3
    point_mat = 4


Op = Operator


@dataclasses.dataclass
class Side:
    """Specifies side of the rig component."""

    left = "l"
    right = "r"
    none = ""


@dataclasses.dataclass
class Ro:
    """Rotate orders."""

    xyz = 0
    yzx = 1
    zxy = 2
    xzy = 3
    yxz = 4
    zyx = 5


@dataclasses.dataclass
class Color:
    """Indices of colors those are used in megarig module."""

    green = 14
    soft_green = 26
    dark_green = 7

    yellow = 17
    soft_yellow = 21
    dark_yellow = 25

    blue = 6
    soft_blue = 18
    dark_blue = 15

    gray = 2
    soft_gray = 3
    dark_gray = 1

    red = 13
    soft_red = 31
    dark_red = 12

    black = 1
    white = 16
    brown = 11
    pink = 20
    none = None


@dataclasses.dataclass
class DrawStyle:
    """Draw style attribute values on joint node."""

    bone = 0
    box = 1
    none = 2


Ds = DrawStyle

Cp = curve_param.CurveParam


class Node:
    """Forward declaration"""

    pass


class Dag:
    """Forward declaration"""

    pass


class Attr(object):
    """For "node.attribute" """

    def __init__(self, attr_str: str):
        tokens = str(attr_str).split(".")

        self.name = attr_str
        self.node = Node(tokens[0])
        self.attr = ".".join(tokens[1:])
        self.query_name = re.findall(r"([_a-zA-Z0-9]+)", tokens[-1])[0]
        self.children = None
        if self.exists:
            self.children = mc.attributeQuery(
                self.query_name, node=self.node, listChildren=True
            )

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __rshift__(self, target: str):
        """Connects current attribute to the target."""
        mc.connectAttr(self, target, f=True)

    def __floordiv__(self, target: str):
        """disconnects current attribute to the target."""
        mc.disconnectAttr(self, target)

    def _setter(self, *args, **kwargs) -> bool:
        """Sets current attribute and its children using mc.setAttr
        with the given args and kwargs.

        If a type of current attribute is typed,
        sets the attribute in a string format.

        If current attribute doesn't have children attrbutes,
        simply use setAttr command.

        If current attribute has children attributes.
        Checks first if no args assigned, sets the attribute properties
        else sets the attribute values.
        """
        if self.type == "typed":
            # In case of current attribtue type is string.
            mc.setAttr(self, *args, type="string", **kwargs)
            return True

        if not self.children:
            mc.setAttr(self, *args, **kwargs)
            return True
        else:
            if not args:
                # Setting properties of attribute, e.g. sets lock and hide.
                for child in self.children:
                    current_attr = "{node_attr}.{child_attr}".format(
                        node_attr=self, child_attr=child
                    )
                    mc.setAttr(current_attr, **kwargs)
            else:
                if len(args[0]) == len(self.children):
                    # Checks the length of given args(values).
                    # It needs to match with the length of children attributes.
                    for sub_val, child in zip(args[0], self.children):
                        current_attr = "{node_attr}.{child_attr}".format(
                            node_attr=self, child_attr=child
                        )
                        mc.setAttr(current_attr, sub_val, **kwargs)
                    return True
                else:
                    raise ValueError(
                        "Mismatch of lengthes between assigning values"
                        + " and children attributes."
                    )

    @property
    def value(self) -> Union[int, float, bool, str]:
        val = mc.getAttr(self)

        if self.type == "typed":
            # For array type attrbute.
            return val

        if isinstance(val, list) or isinstance(val, tuple):
            if isinstance(val[0], list) or isinstance(val[0], tuple):
                val = val[0]
        return val

    @value.setter
    def value(self, val: Union[int, float, bool, str]):
        self._setter(val)

    v = value

    @property
    def lock(self) -> Union[int, float, bool, str]:
        return mc.getAttr(self, lock=True)

    @lock.setter
    def lock(self, val: bool):
        self._setter(lock=val)

    @property
    def hide(self) -> bool:
        return not mc.getAttr(self, k=True)

    @hide.setter
    def hide(self, val: bool):
        self._setter(k=not val)
        self._setter(cb=not val)

    @property
    def type(self) -> str:
        return mc.attributeQuery(
            self.query_name, node=self.node, attributeType=True
        )

    @property
    def exists(self) -> bool:
        return mc.attributeQuery(self.query_name, node=self.node.name, ex=True)

    def connections(self, **kwargs) -> List[Node]:
        cons = mc.listConnections(self, **kwargs)
        if not cons:
            return None
        return [
            Dag(con) if (is_transform(con) or is_shape(con)) else Node(con)
            for con in cons
        ]

    def get_input(self) -> Union["Attr", None]:
        cons = mc.listConnections(self, s=True, d=False, p=True)
        if cons:
            return Attr(cons[0])
        else:
            return None

    def get_output(self) -> Union["Attr", None]:
        """Gets only the first plug of the out plugs.

        Returns:
            Attr/None:
        """
        cons = mc.listConnections(self, s=False, d=True, p=True)
        if cons:
            return Attr(cons[0])
        else:
            return None


class Node(object):
    """Base class for all nodes in megarig"""

    def __init__(self, name: str):
        if not mc.objExists(name):
            raise ValueError("{} does not exist.".format(name))

        self.sel_list = om.MSelectionList()
        self.sel_list.add(name)

        self.m_object = om.MObject()
        self.sel_list.getDependNode(0, self.m_object)

        self.fn_dependency_node = om.MFnDependencyNode(self.m_object)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def name(self):
        return self.fn_dependency_node.name()

    @name.setter
    def name(self, new_name: str):
        mc.rename(self.name, new_name)

    def attr(self, attr_str) -> Attr:
        """Instances Attr object to the given attr_str.

        Args:
            attr_str (str): A name of a node's attribute.

        Returns:
            Attr:
        """
        return Attr("{node}.{attr}".format(node=self, attr=attr_str))

    def add(self, *args, **kwargs) -> Attr:
        """Adds an attribute using addAttr command."""
        attrname = ""
        if "ln" in kwargs:
            attrname = kwargs["ln"]
        elif "longName" in kwargs:
            attrname = kwargs["longName"]

        mc.addAttr(self, *args, **kwargs)

        return self.attr(attrname)

    def lhattrs(self, *args):
        """Locks and hides given attributes."""
        for attr in args:
            self.attr(attr).lock = True
            self.attr(attr).hide = True

    def get_side(self) -> Union[Side, None]:
        """Gets side of the current node.

        Returns:
            Side/None
        """
        if not naming_tools.is_legal(self.name):
            return None

        _, _, side, _ = naming_tools.tokenize(self.name)

        if side == Side.left:
            return Side.left
        elif side == Side.right:
            return Side.right
        else:
            return None


class Dag(Node):
    """Base class for all DAG objects in megarig.
    Casts existed Maya node to megarig Dag nobject.
    """

    def __init__(self, name: str):
        super(Dag, self).__init__(name)

    @property
    def m_dag_path(self) -> om.MDagPath:
        mdagpath = om.MDagPath()
        self.sel_list.getDagPath(0, mdagpath)

        return mdagpath

    @property
    def fn_dag_node(self) -> om.MFnDagNode:
        return om.MFnDagNode(self.m_dag_path)

    @property
    def name(self) -> str:
        if len(mc.ls(self.m_dag_path.partialPathName())) > 1:
            return self.m_dag_path.fullPathName()
        else:
            return self.m_dag_path.partialPathName()

    @name.setter
    def name(self, new_name: str):
        mc.rename(self.name, new_name)

    @property
    def x_axis(self) -> Tuple[float]:
        mat = self.m_dag_path.inclusiveMatrix()
        getarrayitem = om.MScriptUtil.getDoubleArrayItem

        return tuple(getarrayitem(mat[0], ix) for ix in range(3))

    @property
    def y_axis(self) -> Tuple[float]:
        mat = self.m_dag_path.inclusiveMatrix()
        getarrayitem = om.MScriptUtil.getDoubleArrayItem

        return tuple(getarrayitem(mat[1], ix) for ix in range(3))

    @property
    def z_axis(self) -> Tuple[float]:
        mat = self.m_dag_path.inclusiveMatrix()
        getarrayitem = om.MScriptUtil.getDoubleArrayItem

        return tuple(getarrayitem(mat[2], ix) for ix in range(3))

    @property
    def thickness(self) -> float:
        shape = self.get_shape()
        if shape.attr("lineWidth").exists:
            return shape.attr("lineWidth").v

    @thickness.setter
    def thickness(self, v: float):
        shape = self.get_shape()
        if shape.attr("lineWidth").exists:
            shape.attr("lineWidth").v = v

    def create_curve(self, curve_param: Cp) -> "Dag":
        """Creates a curve shape and parent it under
        current transform node.
        """
        shape = self.get_shape()
        if shape:
            print("{xform} already has a shape node.".format(xform=self))
            return shape

        shape = mc.createNode(
            "nurbsCurve", n="{xform}Shape".format(xform=self.name), p=self
        )
        mc.setAttr("{node}.v".format(node=shape), k=False)

        if curve_param == Cp.nurbs_circle:
            mc.setAttr(
                "{s}.cc".format(s=shape),
                3,
                12,
                2,
                False,
                3,
                (-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
                17,
                15,
                (0.5234, 0.0, -0.9065),
                (0.0, 0.0, -1.0467),
                (-0.5234, 0.0, -0.9065),
                (-0.9065, 0.0, -0.5234),
                (-1.0467, 0.0, -0.0),
                (-0.9065, 0.0, 0.5234),
                (-0.5234, 0.0, 0.9065),
                (0.0, 0.0, 1.0467),
                (0.5234, 0.0, 0.9065),
                (0.9065, 0.0, 0.5234),
                (1.0467, 0.0, 0.0),
                (0.9065, 0.0, -0.5234),
                (0.5234, 0.0, -0.9065),
                (0.0, 0.0, -1.0467),
                (-0.5234, 0.0, -0.9065),
                type="nurbsCurve",
            )
        else:
            values = [
                "{s}.cc".format(s=shape),
                1,
                len(curve_param) - 1,
                0,
                False,
                3,
                [ix for ix in range(len(curve_param))],
                len(curve_param),
                len(curve_param),
            ]
            values += curve_param
            mc.setAttr(*values, type="nurbsCurve")

        return shape

    def get_parent(self) -> Union["Dag", None]:
        """Gets a parent of current Dag.

        Returns:
            Dag if parent is a DAG, None if parent is world.
        """
        parent = self.fn_dag_node.parent(0)
        if parent.apiType() == om.MFn.kWorld:
            return None
        parentname = om.MFnDagNode(parent).fullPathName()
        return Dag(parentname)

    def get_shape(self) -> Union["Dag", None]:
        """Gets the first shape of current Dag.

        Returns:
            A megarig Dag of the first shape.
            None if current Dag doesn't have shape node.
        """
        shapes = mc.listRelatives(self, s=True, f=True)
        if not shapes:
            return None
        return Dag(shapes[0])

    def get_all_shapes(self) -> Union[List["Dag"], None]:
        shapes = mc.listRelatives(self, s=True, f=True)
        if not shapes:
            return None
        return [Dag(shape) for shape in shapes]

    def get_orig_shape(self) -> "Dag":
        shapes = mc.listRelatives(self, s=True, f=True)
        for shape in shapes:
            if mc.getAttr(shape + ".intermediateObject"):
                return Dag(shape)
        else:
            return Dag(shapes[0])

    def get_child(self) -> Union["Dag", None]:
        """Gets the first child Dag.

        Returns:
            Dag/None:
        """
        children = mc.listRelatives(self.name, type="transform")
        if children:
            return Dag(children[0])
        else:
            None

    def get_all_children(self) -> List["Dag"]:
        """Gets all child Dags

        Returns:
            List:
        """
        children = mc.listRelatives(self.name, type="transform")
        if children:
            return [Dag(chd) for chd in children]

    def freeze(self, **kwargs):
        mc.makeIdentity(self, a=True, **kwargs)

    def rotate(self, values: Tuple[float], **kwargs):
        mc.rotate(values[0], values[1], values[2], self.name, **kwargs)

    def lhtrs(self):
        """Locks and hides translate, rotate and scale attributes."""
        self.attr("t").lock = True
        self.attr("t").hide = True
        self.attr("r").lock = True
        self.attr("r").hide = True
        self.attr("s").lock = True
        self.attr("s").hide = True

    @property
    def hide(self) -> int:
        return mc.getAttr("{n}.v".format(n=self))

    @hide.setter
    def hide(self, val: bool):
        mc.setAttr("{n}.v".format(n=self), not val)

    @property
    def color(self) -> int:
        """Gets overrideColor of current shape."""
        shape = self.get_shape()
        if not shape:
            return None

        return mc.getAttr("{node}.overrideColor".format(node=shape))

    @color.setter
    def color(self, value: int):
        """Sets override color of current shape node.

        Args:
            value (int): An index color.
        """
        shape = self.get_shape()
        if shape:
            shape.attr("overrideEnabled").value = True
            shape.attr("overrideColor").value = value

    @property
    def rotate_order(self) -> int:
        return mc.getAttr("{node}.rotateOrder".format(node=self))

    @rotate_order.setter
    def rotate_order(self, value: Union[int, Ro]):
        mc.setAttr("{node}.rotateOrder".format(node=self), value)

    @property
    def world_pos(self) -> List[float]:
        return mc.xform(self, q=True, rp=True, ws=True)

    @world_pos.setter
    def world_pos(self, pos: Tuple[float]):
        mc.xform(self, t=pos, ws=True)

    @property
    def world_vec(self) -> om.MVector:
        return om.MVector(*self.world_pos)

    @property
    def ssc(self) -> bool:
        if self.attr("ssc").exists:
            return self.attr("ssc").v

    @ssc.setter
    def ssc(self, value: bool):
        if self.attr("ssc").exists:
            self.attr("ssc").v = value

    # Snapping methods.
    def snap_point(self, target: "Dag"):
        """Snaps position of current DAG object to target."""
        mc.matchTransform(
            self.name, target.name, pos=True, rot=False, scl=False
        )

    def snap_orient(self, target: "Dag"):
        """Snaps orientation of current DAG object to target."""
        mc.matchTransform(
            self.name, target.name, pos=False, rot=True, scl=False
        )

    def snap_scale(self, target: "Dag"):
        """Snaps scale of current DAG object to target."""
        mc.matchTransform(
            self.name, target.name, pos=False, rot=False, scl=True
        )

    def snap(self, target: "Dag"):
        """Snaps transformation of current DAG object to target."""
        mc.matchTransform(self.name, target.name)

    def set_parent(self, parent: "Dag"):
        """Parents current Dag to given parent node."""
        if self.m_object.hasFn(om.MFn.kShape):
            mc.parent(self, parent, s=True, r=True)
        else:
            if parent:
                try:
                    mc.parent(self, parent)
                except RuntimeError:
                    # If the object is already a child of the given parent.
                    pass
            else:
                if self.get_parent():
                    mc.parent(self, w=True)

    def set_pivot(self, pos: Tuple[float]):
        """Sets the rotate and scale pivots to the given pos."""
        mc.move(
            pos[0],
            pos[1],
            pos[2],
            "{}.scalePivot".format(self.name),
            "{}.rotatePivot".format(self.name),
            absolute=True,
        )

    # Shape tools.
    def scale_shape(self, value: Tuple[float]):
        """Scale current shape."""
        compoes = get_all_components(self.name)
        piv = mc.xform(self.name, q=True, rp=True, ws=True)
        mc.scale(value, value, value, compoes, pivot=piv, r=True)

    def rotate_shape(self, values: Tuple[float]):
        """Rotates currnet shape."""
        compoes = get_all_components(self.name)
        mc.rotate(
            values[0],
            values[1],
            values[2],
            compoes,
            relative=True,
            objectSpace=True,
        )

    def move_shape(self, values: Tuple[float]):
        """Moves shape under the object space."""
        compoes = get_all_components(self.name)
        mc.move(
            values[0],
            values[1],
            values[2],
            compoes,
            relative=True,
            objectSpace=True,
            worldSpaceDistance=True,
        )

    # Attribute Tools
    def disable(self):
        """Locks and hides all keyable attributes and hide its shape."""
        for attr in mc.listAttr(self.name, k=True):
            self.attr(attr).lock = True
            self.attr(attr).hide = True

        shape = self.get_shape()
        if shape:
            shape.attr("v").v = False


class Null(Dag):
    """A transform node."""

    def __init__(self):
        xform = mc.createNode("transform", n="null")
        super(Null, self).__init__(xform)


class Locator(Dag):
    """A locator node."""

    def __init__(self):
        xform = mc.spaceLocator()[0]
        super(Locator, self).__init__(xform)


class Joint(Dag):
    """A joint node."""

    def __init__(self):
        xform = mc.createNode("joint")
        super(Joint, self).__init__(xform)

    @property
    def ssc(self) -> bool:
        return self.attr("ssc").v

    @ssc.setter
    def ssc(self, val: bool):
        self.attr("ssc").v = val

    @property
    def radi(self) -> float:
        if self.attr("radi").exists:
            return self.attr("radi").v

    @radi.setter
    def radi(self, value: float):
        if self.attr("radi").exists:
            self.attr("radi").v = value


class Controller(Dag):
    """is a transform node with curve shape"""

    def __init__(self, curve_param: Cp):
        xform = mc.createNode("transform", n="control")
        super(Controller, self).__init__(xform)

        if curve_param:
            self.create_curve(curve_param)

        mc.select(xform, r=True)
        mc.TagAsController()

        self.attr("ro").hide = False


class JointController(Dag):
    """is a joint node with curve shape"""

    def __init__(self, curve_param: Cp):
        xform = mc.createNode("joint", n="jointControl")
        super(JointController, self).__init__(xform)

        if curve_param:
            self.create_curve(curve_param)

        self.attr("drawStyle").v = DrawStyle.none

        for lockedattr in ("radi", "v"):
            self.attr(lockedattr).lock = True
            self.attr(lockedattr).hide = True

        mc.select(xform, r=True)
        mc.TagAsController()

        self.attr("ro").hide = False


class Component(object):
    """A class for vertex, control vertex, control point, uv"""

    def __init__(self, name: str):
        self.node_name, self.compo_name = name.split(".")

        self.sel_list = om.MSelectionList()
        self.sel_list.add(name)

        self.m_object = om.MObject()
        self.m_dag_path = om.MDagPath()
        self.sel_list.getDagPath(0, self.m_dag_path, self.m_object)

        self.fn_comp_node = om.MFnSingleIndexedComponent(self.m_object)

        # Gets component type.
        if self.m_object.hasFn(om.MFn.kMeshVertComponent):
            self.comp_type_str = "vtx"
        elif self.m_object.hasFn(
            om.MFn.kCurveCVComponent
        ) or self.m_object.hasFn(om.MFn.kSurfaceCVComponent):
            self.comp_type_str = "cv"
        else:
            self.comp_type_str = "map"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def name(self) -> str:
        return "{o}.{t}[{i}]".format(
            o=self.m_dag_path.fullPathName(),
            t=self.comp_type_str,
            i=self.fn_comp_node.element(0),
        )

    @property
    def obj_name(self) -> str:
        return self.m_dag_path.fullPathName()

    @property
    def world_pos(self) -> List[float]:
        """Gets the world space coordinates of component"""
        return mc.xform(self.name, q=True, t=True, ws=True)

    @world_pos.setter
    def world_pos(self, t: List[float]) -> List[float]:
        return mc.xform(self.name, t=t, ws=True)

    @property
    def world_vec(self) -> om.MVector:
        return om.MVector(*self.world_pos)

    def get_order(self) -> int:
        allcompoes = [
            compo.split(".")[-1]
            for compo in get_all_components(self.m_dag_path.fullPathName())
        ]
        for ix, compo in enumerate(allcompoes):
            if self.name.split(".")[-1] == compo:
                return ix


class CurveShape(object):
    def __init__(self, curve_shape: Dag) -> None:
        curvefn = om.MFnNurbsCurve(curve_shape.m_dag_path)

        self.name = curvefn.name()
        self.degree = int(curvefn.degree())
        self.form = int(curvefn.form()) - 1
        self.spans = int(curvefn.numSpans())

        knots = om.MDoubleArray()
        curvefn.getKnots(knots)
        self.knots = [int(ix) for ix in knots]

        cvs = om.MPointArray()
        curvefn.getCVs(cvs, om.MSpace.kObject)

        self.cvs = [
            (cvs[ix].x, cvs[ix].y, cvs[ix].z) for ix in range(cvs.length())
        ]

        self.num_knots = curvefn.numKnots()

    def rebuild(self, shape: "CurveShape") -> None:
        values = [
            "{}.cc".format(self.name),
            shape.degree,
            shape.spans,
            shape.form,
            False,
            3,
            shape.knots,
            shape.num_knots,
            len(shape.cvs),
        ]
        values += shape.cvs
        mc.setAttr(*values, type="nurbsCurve")


class Curve(object):
    def __init__(self, curve: Dag) -> None:
        self.name = curve.name
        self.shapes = [CurveShape(s) for s in curve.get_all_shapes()]


def build_curve(curve: Curve) -> Dag:
    dag = create("transform", n=curve.name)

    for shape in curve.shapes:
        shape_name = create("nurbsCurve", n=shape.name, p=dag)

        values = [
            "{}.cc".format(shape_name),
            shape.degree,
            shape.spans,
            shape.form,
            False,
            3,
            shape.knots,
            shape.num_knots,
            len(shape.cvs),
        ]
        values += shape.cvs
        mc.setAttr(*values, type="nurbsCurve")

    return dag


def mobject_from_str(name: str) -> om.MObject:
    sellist = om.MSelectionList()
    sellist.add(str(name))

    mobj = om.MObject()
    sellist.getDependNode(0, mobj)

    return mobj


def dag_path_from_dag_str(name: str) -> om.MDagPath:
    mobj = mobject_from_str(str(name))

    if not mobj.hasFn(om.MFn.kDagNode):
        return None

    dagpath = om.MDagPath().getAPathTo(mobj)

    return dagpath


def dag_path_and_mobject_from_dag_str(
    name: str,
) -> Tuple[om.MDagPath, om.MObject]:
    sellist = om.MSelectionList()
    sellist.add(str(name))

    dagpath = om.MDagPath()
    mobj = om.MObject()

    sellist.getDagPath(0, dagpath, mobj)

    return dagpath, mobj


def is_transform(name: str) -> bool:
    mobj = mobject_from_str(name)

    if mobj.hasFn(om.MFn.kTransform):
        return True
    elif mobj.hasFn(om.MFn.kMesh):
        return True
    elif mobj.hasFn(om.MFn.kNurbsCurve):
        return True
    elif mobj.hasFn(om.MFn.kNurbsSurface):
        return True

    return False


def is_shape(name: str) -> bool:
    mobj = mobject_from_str(name)

    if mobj.hasFn(om.MFn.kShape):
        return True

    return False


def is_attribute(name: str) -> bool:
    sellist = om.MSelectionList()
    sellist.add(name)
    mplug = om.MPlug()

    try:
        sellist.getPlug(0, mplug)
        return True
    except RuntimeError:
        return False


def is_component(name: str) -> bool:
    sellist = om.MSelectionList()
    sellist.add(name)
    dagpath = om.MDagPath()
    mobj = om.MObject()
    sellist.getDagPath(0, dagpath, mobj)

    if mobj.hasFn(om.MFn.kComponent):
        return True

    return False


def to_dags(names: list) -> List[Dag]:
    return tuple([Dag(name) for name in names])


def to_components(names: list) -> list:
    return tuple([Component(c) for c in mc.ls(names, fl=True)])


def create(*args, **kwargs) -> Union[Dag, Node]:
    node = mc.createNode(*args, **kwargs)
    if is_transform(node):
        return Dag(node)
    else:
        return Node(node)


def group(*args, **kwargs) -> Dag:
    grp = Dag(mc.group(*args, **kwargs))
    grp.attr("rp").v = (0, 0, 0)
    grp.attr("sp").v = (0, 0, 0)

    return grp


def cluster(*args, **kwargs) -> Tuple[Node, Dag]:
    clust, clust_handle = mc.cluster(*args, **kwargs)

    return Node(clust), Dag(clust_handle)


def aimcons(*args, **kwargs) -> Dag:
    return Dag(mc.aimConstraint(*args, **kwargs)[0])


def oricons(*args, **kwargs) -> Dag:
    cons = Dag(mc.orientConstraint(*args, **kwargs)[0])
    cons.attr("interpType").v = 2
    return cons


def pntcons(*args, **kwargs) -> Dag:
    return Dag(mc.pointConstraint(*args, **kwargs)[0])


def parcons(*args, **kwargs) -> Dag:
    cons = Dag(mc.parentConstraint(*args, **kwargs)[0])
    cons.attr("interpType").v = 2
    return cons


def scacons(*args, **kwargs) -> Dag:
    return Dag(mc.scaleConstraint(*args, **kwargs)[0])


def pscons(*args, **kwargs) -> Tuple[Dag, Dag]:
    parcons = Dag(mc.parentConstraint(*args, **kwargs)[0])
    scacons = Dag(mc.scaleConstraint(*args, **kwargs)[0])

    return parcons, scacons


def get_all_components(name: str) -> List[str]:
    """Returns a list of tokenized component names."""
    sel_list = om.MSelectionList()
    sel_list.add(name)
    dag_path = om.MDagPath()
    sel_list.getDagPath(0, dag_path)

    dag_path.extendToShape()

    compoes = None
    # Mesh
    if dag_path.hasFn(om.MFn.kMesh):
        mesh_fn = om.MFnMesh(dag_path)
        num_vertices = mesh_fn.numVertices()
        compoes = "{dp}.vtx[0:{vtx}]".format(
            dp=dag_path.partialPathName(), vtx=num_vertices - 1
        )
    # NURBs Curve
    elif dag_path.hasFn(om.MFn.kNurbsCurve):
        curve_fn = om.MFnNurbsCurve(dag_path)
        spans = curve_fn.numSpans()
        degree = curve_fn.degree()
        form = curve_fn.form()

        num_cvs = spans + degree

        if form == 3:
            num_cvs -= degree

        compoes = "{dp}.cv[0:{cv}]".format(
            dp=dag_path.partialPathName(), cv=num_cvs - 1
        )
    # NURBs Surface
    elif dag_path.hasFn(om.MFn.kNurbsSurface):
        nurbs_fn = om.MFnNurbsSurface(dag_path)

        spans_u = nurbs_fn.numSpansInU()
        degree_u = nurbs_fn.degreeU()

        spans_v = nurbs_fn.numSpansInV()
        degree_v = nurbs_fn.degreeV()

        form_u = nurbs_fn.formInU()
        form_v = nurbs_fn.formInV()

        num_u = spans_u + degree_u

        if form_u == 3:
            num_u -= degree_u

        num_v = spans_v + degree_v

        if form_v == 3:
            num_v -= degree_v

        compoes = "{dp}.cv[0:{u}][0:{v}]".format(
            dp=dag_path.partialPathName(), u=num_u - 1, v=num_v - 1
        )
    # Lattice
    elif dag_path.hasFn(om.MFn.kLattice):
        latname = dag_path.partialPathName()
        ns = mc.getAttr("{}.sDivisions".format(latname))
        nt = mc.getAttr("{}.tDivisions".format(latname))
        nu = mc.getAttr("{}.uDivisions".format(latname))
        compoes = "{dp}.pt[0:{ns}][0:{nt}][0:{nu}]".format(
            dp=latname, ns=ns, nt=nt, nu=nu
        )

    return mc.ls(compoes, fl=True)


def get_compo_index(compo_str: str) -> int:
    """Returns an ordered component index."""
    for ix, compo in enumerate(get_all_components(compo_str.split(".")[0])):
        if compo.split(".")[-1] == compo_str.split(".")[-1]:
            return ix


def point_constraint(*args, **kwargs) -> Dag:
    return Dag(mc.pointConstraint(*args, **kwargs)[0])


def parent_constraint(*args, **kwargs) -> Dag:
    cons = Dag(mc.parentConstraint(*args, **kwargs)[0])
    cons.attr("interpType").v = 2
    return cons


def scale_constraint(*args, **kwargs) -> Dag:
    return Dag(mc.scaleConstraint(*args, **kwargs)[0])


def ps_constraint(*args, **kwargs) -> Tuple[Dag, Dag]:
    parcons = Dag(mc.parentConstraint(*args, **kwargs)[0])
    scacons = Dag(mc.scaleConstraint(*args, **kwargs)[0])

    return parcons, scacons


def aim_constraint(*args, **kwargs) -> Dag:
    return Dag(mc.aimConstraint(*args, **kwargs)[0])
