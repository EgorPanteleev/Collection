Colors = {
    WHITE     = { 255, 255, 255 },
    ORANGE    = { 255, 127, 0   },
    GRAY      = { 210, 210, 210 },
    RED       = { 255, 0  , 0   },
    GREEN     = { 0  , 255, 0   },
    BLUE      = { 0  , 0  , 255 },
    YELLOW    = { 255, 255, 0   },
    BROWN     = { 150, 75 , 0   },
    PINK      = { 255, 105, 180 },
    DARK_BLUE = { 65 , 105, 225 },
    CYAN      = { 0  , 255, 255 }
}

Camera = {
	pos = { 0, 10, 0 },
	lookAt = { 0, 0, 1 },
	FOV = 67.38,
}

Canvas = {
	w = 800,
	h = 500,
}

Objects = {
    Meshes = {
        --right
        {
            Type = "CubeMesh",
            min = { 70, -50, 0 },
            max = { 80, 70, 600 },
            material = { Colors.GREEN, -1, 0 }
        },
        --left
        {
            Type = "CubeMesh",
            min = { -80, -50, 0 },
            max = { -70, 70, 600 },
            material = { Colors.RED, -1, 0 }
        },
        --front
        {
            Type = "CubeMesh",
            min = { -100, -50, 290 },
            max = { 100, 70, 300 },
            material = { Colors.GRAY, -1, 0 }
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
    }
}

Lights = {
    Spheres = {
        {
            origin = { 0, 50, 150 },
            radius = 15,
            material = { Colors.DARK_BLUE, 0.7 }
        },
    },
--    Meshes = {
--        -- Define mesh lights here
--    }
}

RenderSettings = {
    depth = 1,
    ambientSamples = 5,
    lightSamples = 5,
    denoise = 1,
}