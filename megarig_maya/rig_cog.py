from megarig_maya import core as mrc
from megarig_maya import rig_base
from megarig_maya import utils as mru
from megarig_maya import rig_main_group


class CogRig(rig_base.BaseRig):
    def __init__(
        self,
        guide: str,
        desc: str,
        main_grp: rig_main_group.MainGroup,
    ):
        self.guide = mrc.Dag(guide)

        super(CogRig, self).__init__("cog", desc, mrc.Side.none)

        (
            self.ctrl,
            self.ofst,
            self.ori,
            self.zr,
        ) = self.add_controller(
            mrc.Side.none,
            ("",),
            mrc.Cp.double_circle,
            mrc.Color.yellow,
            ["s", "v"],
        )
        self.ctrl.rotate_order = mrc.Ro.xzy
        self.zr.snap(self.guide)
        self.zr.set_parent(self.meta)

        # Skin Joint
        self.jnt = self.init_dag(
            mru.create_joint_at(self.guide), mrc.Side.none, ("",), "jnt"
        )
        cons = mrc.parent_constraint(self.ctrl, self.jnt)
        cons.set_parent(self.meta)
        self.jnt.set_parent(main_grp.local_offset_ctrl)
        self.skin_jnts = [self.jnt]

        self.meta.set_parent(main_grp.local_offset_ctrl)
        # Skin Joints
        self.add_skin_joints_to([self.jnt], mrc.SKIN_SET, main_grp.jnt_vis)
