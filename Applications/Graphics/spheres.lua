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
    CYAN      = { 0  , 255, 255 }
}

Camera = {
	pos = { 0, 0, -10000 },
	lookAt = { 0, 0, 1 },
	FOV = 33,
}

Canvas = {
	w = 3200,
	h = 2000,
}

Objects = {
    Spheres = {
        {
            origin = { 0, 0, 3000 },
            radius = 1500,
            material = { Colors.ORANGE, -1, 0.5 }
        },
        {
            origin = { 2121, 0, 2250 },
            radius = 300,
            material = { Colors.RED, 500  }
        },
        {
            origin = { 1030, 0, 1000 },
            radius = 300,
            material = { Colors.BLUE, -1, 0 }
        },
        {
            origin = { -2121, 0, 2250 },
            radius = 300,
            material = { Colors.CYAN, 500 }
        },
        {
            origin = { -1030, 0, 1000 },
            radius = 300,
            material = { Colors.PINK, -1, 0 }
        },
    },
--     Meshes = {
--         {
--             Type = "CubeMesh",
--             min = { 0, 0, 300 },
--             max = { 500, 500, 1000 },
--             material = { Colors.YELLOW, -1, 0 }
--         }
-- --         {
-- --             Type = "BaseMesh",
-- --             Path = "",
-- --             material = { Colors.YELLOW, -1, 0 }
-- --         },
--     }
}

Lights = {
    Spheres = {
--         {
--             origin = { -1030, 0, 1000 },
--             radius = 300,
--             material = { Colors.PINK, 500 }
--         },
--         {
--             origin = { 0, 0, 3000 },
--             radius = 1500,
--             material = { Colors.ORANGE, 500 }
--         },
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