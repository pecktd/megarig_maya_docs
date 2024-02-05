# pkrig modules
from megarig_maya import core as mrc
from megarig_maya import rig_base
from megarig_maya import utils as mru


class MainGroup(rig_base.BaseRig):
    def __init__(self, asset_name: str):
        if not asset_name:
            asset_name = "rig"

        super(MainGroup, self).__init__(asset_name, "", mrc.Side.none)

        self.mod_name = ""
        self.mod_desc = ""

        self.world_ctrl = self.init_dag(
            mrc.Controller(mrc.Cp.square), mrc.Side.none, ("world",), "ctrl"
        )
        self.world_ctrl.color = mrc.Color.green
        self.world_ctrl.lhattrs("v")
        self.world_ctrl.scale_shape(2.5)
        self.world_ctrl.set_parent(self.meta)

        def _add_sub_ctrl(
            name: str,
            parent: mrc.Dag,
            shape: mrc.Cp,
            scale: float,
            color: mrc.Color,
        ) -> mrc.Dag:
            ctrl = self.init_dag(
                mrc.Controller(shape), mrc.Side.none, (name,), "ctrl"
            )
            ctrl.color = color
            ctrl.scale_shape(scale)
            ctrl.lhattrs("s", "v")
            ctrl.set_parent(parent)

            return ctrl

        self.global_ctrl = _add_sub_ctrl(
            "global", self.world_ctrl, mrc.Cp.square, 2, mrc.Color.dark_green
        )
        self.local_ctrl = _add_sub_ctrl(
            "local", self.global_ctrl, mrc.Cp.square, 1.5, mrc.Color.soft_green
        )
        self.local_offset_ctrl = _add_sub_ctrl(
            "local_offset", self.local_ctrl, mrc.Cp.circle, 1, mrc.Color.yellow
        )

        self.still_grp = self.init_dag(
            mrc.Null(), mrc.Side.none, ("still",), "grp"
        )
        self.still_grp.set_parent(self.meta)
        self.still_grp.lhtrs()

        self.skin_set = self.init_node(
            mrc.create("objectSet"), mrc.Side.none, ("skin",), "set"
        )

        # Sets default values
        self.still_grp.attr("v").v = False

        # Vis Attrs
        mru.add_divide_attr(self.world_ctrl, "vis")
        jnt_vis_attr = self.world_ctrl.add(
            ln="jntVis", dv=True, at="bool", k=True
        )

        self.jnt_vis_cond = self.init_node(
            mrc.create("condition"), mrc.Side.none, ("jnt_vis",), "cond"
        )
        self.jnt_vis_cond.attr("op").v = mrc.Operator.equal
        jnt_vis_attr >> self.jnt_vis_cond.attr("ft")
        self.jnt_vis_cond.attr("st").v = 1
        self.jnt_vis_cond.attr("ctr").v = 0
        self.jnt_vis_cond.attr("cfr").v = 2

        self.jnt_vis = self.jnt_vis_cond.attr("ocr")
