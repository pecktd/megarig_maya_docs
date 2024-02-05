# System Modules
from typing import Union, List, Tuple

# Maya modules
import maya.cmds as mc

# megarig modules
from megarig_maya import core as mrc
from megarig_maya import utils as mru
from megarig_maya import naming_tools as mrnt


META_ATTR_NAME = "_megarig_meta"
SIDE_ATTR_NAME = "moduleSide"
MOD_ATTR_NAME = "moduleName"
DESC_ATTR_NAME = "moduleDesc"


class BaseRig(object):
    """A base class to all rigging modules.
    A BaseRig object is initialized with a group as a Meta Node.
    """

    def __init__(self, mod_name: str, mod_desc: str, side: mrc.Side):
        self.meta = mrc.Null()
        self.mod_name = mod_name
        self.mod_desc = mod_desc
        self.side = side

        self.meta.name = mrnt.compose(side, (self.name,), "grp")

        metaattr = self.meta.add(ln=META_ATTR_NAME, dt="string")
        metaattr.lock = True

        self._add_meta_attrs(self.meta)

    @property
    def name(self) -> str:
        return "_".join([n for n in [self.mod_name, self.mod_desc] if n])

    def current_names(self, names: Tuple[str]) -> Tuple[str]:
        """Insert mod_name and desc_name to the name list for compositing name"""
        return tuple([self.name] + list(names))

    def _add_meta_attrs(self, node: Union[mrc.Node, mrc.Dag]):
        if not node.attr(SIDE_ATTR_NAME).exists:
            sideattr = node.add(ln=SIDE_ATTR_NAME, dt="string")
            sideattr.v = self.side
            sideattr.lock = True

        if not node.attr(MOD_ATTR_NAME).exists:
            modattr = node.add(ln=MOD_ATTR_NAME, dt="string")
            modattr.v = self.mod_name
            modattr.lock = True

        if not node.attr(DESC_ATTR_NAME):
            descattr = node.add(ln=DESC_ATTR_NAME, dt="string")
            descattr.v = self.mod_desc
            descattr.lock = True

    def init_node(
        self,
        node: mrc.Node,
        side: mrc.Side,
        names: Tuple[str],
        type_: str,
    ) -> mrc.Node:
        node.name = mrnt.compose(side, self.current_names(names), type_)
        self._add_meta_attrs(node)

        return node

    def init_dag(
        self,
        dag: mrc.Dag,
        side: mrc.Side,
        names: Tuple[str],
        type_: str,
    ) -> mrc.Dag:
        dag.name = mrnt.compose(side, self.current_names(names), type_)
        self._add_meta_attrs(dag)

        return dag

    def add_controller(
        self,
        side: mrc.Side,
        names: Tuple[str],
        shape: mrc.Cp,
        color: mrc.Color,
        locked_attrs: Tuple[str],
    ) -> Tuple[mrc.Dag, mrc.Dag, mrc.Dag, mrc.Dag]:
        """Adds a controller with its groups

        Returns:
            mrc.Dag: Controller
            mrc.Dag: Offset group
            mrc.Dag: Orient group
            mrc.Dag: Zero group
        """
        ctrl = self.init_dag(mrc.Controller(shape), side, names, "ctrl")
        ctrl.lhattrs(*locked_attrs)
        ctrl.color = color

        ofst = self.init_dag(mrc.group(ctrl), side, names, "ofst")
        ori = self.init_dag(mrc.group(ofst), side, names, "ori")
        zr = self.init_dag(mrc.group(ori), side, names, "zr")

        for grp in (ofst, ori, zr):
            grp.lhattrs("s", "v")

        return ctrl, ofst, ori, zr

    def add_ik_controller(
        self,
        side: mrc.Side,
        names: Tuple[str],
        shape: mrc.Cp,
        color: mrc.Color,
        locked_attrs: List[str],
        target: mrc.Dag,
    ) -> Tuple[mrc.Dag, mrc.Dag, mrc.Dag, mrc.Dag]:
        """IK controller is a JointController if the target is given,
        zero group position is set at the target and the controller
        is oriented to the target.

        Returns:
            mrc.Dag: Controller
            mrc.Dag: Offset group
            mrc.Dag: Orient group
            mrc.Dag: Zero group
        """
        ctrl = self.init_dag(mrc.JointController(shape), side, names, "ctrl")
        ctrl.color = color

        ofst = self.init_dag(mrc.group(ctrl), side, names, "ofst")
        ori = self.init_dag(mrc.group(ofst), side, names, "ori")
        zr = self.init_dag(mrc.group(ori), side, names, "zr")

        if target:
            zr.snap_point(target)
            ctrl.snap_orient(target)
            ctrl.freeze(r=True, s=False, t=False)

        ctrl.lhattrs(*locked_attrs)

        for grp in (ofst, ori, zr):
            grp.lhattrs("s", "v")

        return ctrl, ofst, ori, zr

    def add_curve_guide(
        self,
        root: mrc.Dag,
        end: mrc.Dag,
        side: mrc.Side,
        names: Tuple[str],
    ) -> Tuple[mrc.Dag, mrc.Node, mrc.Dag, mrc.Node, mrc.Dag]:
        """Creates a NURBs curve guide from the root to the end."""
        (
            curve,
            root_cluster,
            root_cluster_handle,
            end_cluster,
            end_cluster_handle,
        ) = mru.curve_guide(root, end)

        curve = self.init_dag(curve, side, names, "crv")
        root_cluster = self.init_node(root_cluster, side, names, "cluster")
        end_cluster = self.init_node(end_cluster, side, names, "cluster")

        return (
            curve,
            root_cluster,
            root_cluster_handle,
            end_cluster,
            end_cluster_handle,
        )

    def add_space_blender(
        self,
        target_nodes_and_attr_names: Tuple[Tuple[mrc.Dag, str]],
        cons_func: Union[mrc.pntcons, mrc.oricons, mrc.parcons],
        constrained_node: mrc.Dag,
        controller: mrc.Dag,
    ) -> Tuple[List[mrc.Dag], List[mrc.Dag], List[mrc.Node], List[str]]:
        """Adds space blender.
        Refer to pru.add_space_blender for the usage.

        Args:
            target_nodes_and_attr_names (tuple):
            cons_func (mrc.orient_constraint/mrc.point_constraint/
                mrc.parent_constraint):
            constrained_node (mrc.Dag):
            controller (mrc.Dag):
            pivot (mrc.Dag):
        """
        spc_nodes, spc_pivs, spc_conds, attrs = mru.add_space_blender(
            target_nodes_and_attr_names,
            cons_func,
            constrained_node,
            controller,
        )

        side, names, _ = mrnt.tokenize(controller.name)
        for spcnode, spccond, attr in zip(spc_nodes, spc_conds, attrs):
            spcname = "_".join(names + [attr])
            self.init_dag(spcnode, side, (spcname,), "grp")
            self.init_node(spccond, side, (spcname,), "cond")

        return spc_nodes, spc_pivs, spc_conds, attrs

    def add_skin_joints_to(
        self,
        jnts: List[mrc.Joint],
        set_name: str,
        jnt_vis_attr: Union[mrc.Attr, None],
    ) -> None:
        """Adds all nodes in skin_jnts attribute to the given set name."""
        mc.select(cl=True)
        if not mc.objExists(set_name):
            mc.sets(n=set_name)
        mc.sets(jnts, add=set_name)

        if jnt_vis_attr:
            for jnt in jnts:
                jnt_vis_attr >> jnt.attr("ds")
