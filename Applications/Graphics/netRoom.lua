Colors = {
    WHITE     = { 255, 255, 255 },
    ORANGE    = { 249, 231, 132 },
    GRAY      = { 210, 210, 210 },
    RED       = { 255, 0  , 0   },
    GREEN     = { 0  , 255, 0   },
    BLUE      = { 0  , 0  , 255 },
    YELLOW    = { 255, 255, 0   },
    BROWN     = { 150, 75 , 0   },
    PINK      = { 255, 105, 180 },
    DARK_BLUE = { 65 , 105, 225 },
    CYAN      = { 0  , 255, 255 },
    LIGHTBLUE  = { 173, 216, 230 }
}

Camera = {
	pos = { 0, 10, 0 },
	lookAt = { 0, 0, 1 },
	FOV = 67.38,
}

Canvas = {
	w = 3200,
	h = 2000,
}

Objects = {
    Meshes = {
        --right
        {
            Type = "CubeMesh",
            min = { 70, -50, 0 },
            max = { 80, 70, 600 },
            material = { Colors.PINK, -1, 0 }
        },
        --left
        {
            Type = "CubeMesh",
            min = { -80, -50, 0 },
            max = { -70, 70, 600 },
            material = { Colors.PINK, -1, 0 }
        },
        --front
        {
            Type = "CubeMesh",
            min = { -100, -50, 290 },
            max = { 100, 70, 300 },
            material = { Colors.LIGHTBLUE, 5 }
        },
        --back
        {
            Type = "CubeMesh",
            min = { -100, -50, -10 },
            max = { 100, 70, 0 },
            material = { Colors.GRAY, -1, 0 }
        },
        --down
        {
            Type = "CubeMesh",
            min = { -100, -70, 0 },
            max = { 100, -50, 620 },
            material = { Colors.GRAY, -1, 0 }
        },
        --up
        {
            Type = "CubeMesh",
            min = { -100, 70, 0 },
            max = { 100, 90, 620 },
            material = { Colors.GRAY, -1, 0 }
        },
        --rand1
        {
            Type = "CubeMesh",
            min = { -15, -50, 310 },
            max = { 15, -30, 340 },
            material = { Colors.DARK_BLUE, -1, 0.7 },
            moveTo = { 0, -40, 325 },
            scaleTo = { 30, 100, 30 },
            rotate = { 0, 25, 0 },
            move = { 20, 0, -150 },
        },
        --rand2
        {
            Type = "CubeMesh",
            min = { -15, -50, 310 },
            max = { 15, -30, 340 },
            material = { Colors.PINK, -1, 1 },
            moveTo = { 0, -40, 325 },
            scaleTo = { 30, 260, 30 },
            rotate = { 0, -25, 0 },
            move = { -15, 0, -100 },
        },
    }
}

Lights = {
    Spheres = {
--         {
--             origin = { 0, 55, 150 },
--             radius = 5,
--             material = { Colors.PINK, 0.5 }
--         },
    },
--    Meshes = {
--        -- Define mesh lights here
--    }
}

RenderSettings = {
    depth = 1,
    ambientSamples = 2,
    lightSamples = 2,
    denoise = 1,
}