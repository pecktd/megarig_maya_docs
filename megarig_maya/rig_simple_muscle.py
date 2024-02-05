from typing import Union

from megarig_maya import core as mrc
from megarig_maya import rig_base
from megarig_maya import utils as mru


class SimpleMuscle(rig_base.BaseRig):
    """A single two joints chain aims to the given target,
    used as the fake muscle"""

    def __init__(
        self,
        parent: Union[mrc.Dag, None],
        root_node: mrc.Dag,
        up_axis: tuple,
        target: mrc.Dag,
        mod_name: str,
        desc: str,
        side: mrc.Side,
    ):
        super(SimpleMuscle, self).__init__(mod_name, desc, side)

        if parent:
            self.meta.set_parent(parent)

        self.length = mru.distance(root_node, target)

        self.root_jnt = self.init_dag(mrc.Joint(), side, ("root",), "jnt")
        self.root_jnt.set_parent(self.meta)
        self.tip_jnt = self.init_dag(mrc.Joint(), side, ("tip",), "jnt")
        self.tip_jnt.set_parent(self.root_jnt)
        self.tip_jnt.attr("ty").v = self.length

        self.meta.snap(root_node)

        self.root_tgt = self.init_dag(mrc.Null(), side, ("root",), "posTgt")
        self.root_tgt.snap(self.meta)
        self.root_tgt.set_parent(root_node)
        mrc.point_constraint(self.root_tgt, self.meta)
        mrc.aim_constraint(
            target,
            self.meta,
            aim=(0, 1, 0),
            u=up_axis,
            wu=up_axis,
            wut="objectrotation",
        )

        def _create_pos_node(pos_target: mrc.Dag, name: str) -> mrc.Dag:
            pos = self.init_dag(mrc.Null(), side, (name,), "pos")
            pos.set_parent(self.meta)
            mrc.point_constraint(pos_target, pos)

            return pos

        self.root_pos = _create_pos_node(root_node, "root")
        self.tip_pos = _create_pos_node(target, "tip")
        self.dist = self.init_node(
            mrc.create("distanceBetween"), side, ("",), "dist"
        )

        self.root_pos.attr("t") >> self.dist.attr("p1")
        self.tip_pos.attr("t") >> self.dist.attr("p2")

        self.scale = self.init_node(
            mrc.create("multiplyDivide"), side, ("scale",), "div"
        )
        self.scale.attr("op").v = mrc.Operator.divide
        self.dist.attr("d") >> self.scale.attr("i1x")
        self.scale.attr("i2x").v = self.length
        self.scale.attr("ox") >> self.root_jnt.attr("sy")
