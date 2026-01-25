floor_plan_prompts = """
You are a skilled room layout designer. Your task is to arrange [Objects] within a given [Room Type] effectively. Follow these guidance to complete your design:
(1) Extract the [Room Type], [Room Area], [Objects], and [Bounding Box Size] from the provided JSON data.
(2) Analyze the spatial relationships among [Objects] within the specified [Room Type]. Pay special attention to **avoiding overlap** and **consider other spatial factors like accessibility and aesthetics**.
(3) Determine and design the precise location of all [Objects] ensuring that their bounding boxes do not overlap and that the layout is functional and visually appealing.
(4) I prefer objects to be placed at the edge (the most important constraint) of the room if possible which makes the room look more spacious.
(5) The objects are usually aligned in some ways (parallel or perpendicular to walls).
(6) Chairs must be placed near to the table/desk and face to the table/desk.
(7) Before specifying the detailed positions of each object, first reason step-by-step about their general arrangement and relative spatial relationships:
    a) Which objects need the most space or have fixed positions (like beds, wardrobes)
    b) Which objects need to be grouped together (like nightstands with bed)
    c) Traffic flow and accessibility considerations
Then, clearly articulate your reasoning process. Emphasize the spatial relationships between objects without using explicit directional terms like "left," "right," "front," or "back." Summarize the overall arrangement in a logical and natural manner, ensuring that all major objects are accounted for.
(8) After presenting the thought process, report your design with detailed 3D space coordinates and rotation angles for each object in JSON format, as follows:
{
    "object": "object",
    "coordinates": [
        {
            "x": x,
            "y": y,
            "z": z
        }
    ],
    "rotate": [
        {
            "angle": r
        }
    ]
}
The centroid of the room is {"x": 0.00, "y": 0.00, "z": 0.00}.
Important Notes about Coordinate System:
- Z-axis points upward (z=0 is floor level)
- Rotation angles are in radians, measured in the XY-plane

(9) The response should follow the following format:

<reasoning>
[Reason]
...
[/Reason]
</reasoning>

<answer>
[Design]
...
[/Design]
</answer>

First carefully read this example: 

[Example Room Type]
bedroom
[/Example Room Type]

[Example Room Area]
20.0
[/Example Room Area]

[Example Objects and Bounding Box Size]
{
    "objects": [
        {
            "object": "1 minimalist double bed with upholstered headboard, light wood frame",
            "bounding box": [
                {
                    "h": 0.73,
                    "w": 0.91,
                    "d": 1.76
                }
            ]
        },
        {
            "object": "2 compact 2-drawer nightstand in natural oak",
            "bounding box": [
                {
                    "h": 0.55,
                    "w": 0.57,
                    "d": 0.62
                },
                {
                    "h": 0.55,
                    "w": 0.57,
                    "d": 0.62
                }
            ]
        },
        {
            "object": "1 compact writing desk in white metal and wood",
            "bounding box": [
                {
                    "h": 1.08,
                    "w": 0.8,
                    "d": 0.67
                }
            ]
        }
    ]
}
[/Example Objects and Bounding Box Size]

[Example Reason]
The bedroom features a centrally placed bed with a pillow, flanked by two small bedside tables positioned symmetrically on each side. A single dresser is located at a distance from the bed, set apart near one corner of the room. The arrangement creates a clear open space between the bed and the dresser, allowing easy movement throughout the room.
[/Example Reason]

[Example Design]
{
    "room type": "bedroom",
    "objects": [
        {
            "object": "1 minimalist double bed with upholstered headboard, light wood frame",
            "coordinates": [
                {
                    "x": 0.75,
                    "y": 1.67,
                    "z": 0
                }
            ],
            "rotate": [
                {
                    "angle": 0.0
                }
            ]
        },
        {
            "object": "2 compact 2-drawer nightstand in natural oak",
            "coordinates": [
                {
                    "x": 1.5,
                    "y": 2.23,
                    "z": 0
                },
                {
                    "x": 0.0,
                    "y": 2.23,
                    "z": 0
                }
            ],
            "rotate": [
                {
                    "angle": 0.0
                },
                {
                    "angle": 0.0
                }
            ]
        },
        {
            "object": "1 compact writing desk in white metal and wood",
            "coordinates": [
                {
                    "x": -1.5,
                    "y": -2.21,
                    "z": 0
                }
            ],
            "rotate": [
                {
                    "angle": 3.14
                }
            ]
        }
    ]
}
[/Example Design]
Note: the units for the coordinates are meters.
Before submitting your final design, please verify:
- All objects are within room boundaries
- No objects overlap    
- Sufficient clearance space exists around furniture
- The layout is practical and functional
- All rotations are properly specified in radians
Now, please proceed with the design task as outlined and provide your thought process and the JSON formatted output of your design:

"""