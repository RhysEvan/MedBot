class trigger():
    def __init__(self, main):
        main.homing.clicked.connect(main.back.main_home)
        main.submit.clicked.connect(main.back.append_coord)
        main.recording.clicked.connect(main.back.append_motor)
        main.remove_motor.clicked.connect(main.back.handle_lists)
        main.animating.clicked.connect(main.back.animation_seq)
        main.path.clicked.connect(main.back.dynamic.show_path)
        main.coord_query.clicked.connect(main.back.query)
        main.robot_options.clicked.connect(main.back.dynamic.clicked)
        main.compiling.clicked.connect(main.back.json_saving)
        main.executing.clicked.connect(main.back.json_choosing)
        
        main.aabs.valueChanged.connect(main.back.dynamic.slider_motor)
        main.babs.valueChanged.connect(main.back.dynamic.slider_motor)
        main.cabs.valueChanged.connect(main.back.dynamic.slider_motor)
        main.dabs.valueChanged.connect(main.back.dynamic.slider_motor)
        main.eabs.valueChanged.connect(main.back.dynamic.slider_motor)
        main.fabs.valueChanged.connect(main.back.dynamic.slider_motor)

        main.End_X.valueChanged.connect(main.back.dynamic.slider_end)
        main.End_Y.valueChanged.connect(main.back.dynamic.slider_end)
        main.End_Z.valueChanged.connect(main.back.dynamic.slider_end)
        main.End_al.valueChanged.connect(main.back.dynamic.slider_end)
        main.End_bt.valueChanged.connect(main.back.dynamic.slider_end)
        main.End_gm.valueChanged.connect(main.back.dynamic.slider_end)

        main.xcoord.textEdited.connect(main.back.dynamic.location)
        main.ycoord.textEdited.connect(main.back.dynamic.location)
        main.zcoord.textEdited.connect(main.back.dynamic.location)
        main.alphacoord.textEdited.connect(main.back.dynamic.location)
        main.betacoord.textEdited.connect(main.back.dynamic.location)
        main.gammacoord.textEdited.connect(main.back.dynamic.location)

        main.DH_param_1.textEdited.connect(main.back.dynamic.change_alpha)
        main.DH_param_2.textEdited.connect(main.back.dynamic.change_theta)
        main.DH_param_3.textEdited.connect(main.back.dynamic.change_radius)
        main.DH_param_4.textEdited.connect(main.back.dynamic.change_dists)
        main.DH_param_5.textEdited.connect(main.back.dynamic.change_active)
        main.DH_param_6.textEdited.connect(main.back.dynamic.change_limits)
        main.Save_DH_param.clicked.connect(main.back.dynamic.json_type)
        main.Push_Changes.clicked.connect(main.back.dynamic.update_visual)
 