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

        main.aabs.valueChanged.connect(main.back.dynamic.slider_input)
        main.babs.valueChanged.connect(main.back.dynamic.slider_input)
        main.cabs.valueChanged.connect(main.back.dynamic.slider_input)
        main.dabs.valueChanged.connect(main.back.dynamic.slider_input)
        main.eabs.valueChanged.connect(main.back.dynamic.slider_input)
        main.fabs.valueChanged.connect(main.back.dynamic.slider_input)
        main.xcoord.textEdited.connect(main.back.dynamic.location)
        main.ycoord.textEdited.connect(main.back.dynamic.location)
        main.zcoord.textEdited.connect(main.back.dynamic.location)
        main.alphacoord.textEdited.connect(main.back.dynamic.location)
        main.betacoord.textEdited.connect(main.back.dynamic.location)
        main.gammacoord.textEdited.connect(main.back.dynamic.location)
 