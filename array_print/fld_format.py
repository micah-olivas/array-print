"""
Headers and footers that make a .fld file readable by the Scienion.

A bare list of positions is not enough: the instrument expects a header
describing the device geometry and a footer holding an encrypted timestamp, the
whole file in CRLF with cp1252 encoding.
"""

HEADERS = {
    "PS1.8K": "\r\n".join(
        [
            "Comment: ",
            "Field(s): ",
            "X = 1",
            "Y = 1",
            "Start Point",
            "Left: 29000",
            "Up: 2400",
            "X Field Gap:2345/",
            "Y Field Gap:900/",
            "Pattern Size: ",
            "X = 56",
            "Y = 32",
            "Dot Pitch: ",
            "X = 375",
            "Y = 655",
            "Type: No. of Drops",
            "Line Spotting:No",
            "Direction = Y",
            "Spot Pitch = 1000",
            "Frequency = 10",
            "Volume (µl/cm) = 0.100000",
            "Type = Volume",
            "Field Data: ",
            "[0, 0, 0]",
        ]
    ),
    "PS1.8K_Premiere": "\r\n".join(
        [
            "Comment: ",
            "Field(s): ",
            "X = 1",
            "Y = 1",
            "Start Point",
            "Left: 20000",
            "Up: 2400",
            "X Field Gap:2345/",
            "Y Field Gap:900/",
            "Pattern Size: ",
            "X = 56",
            "Y = 32",
            "Dot Pitch: ",
            "X = 375",
            "Y = 655",
            "Type: No. of Drops",
            "Line Spotting:No",
            "Direction = Y",
            "Spot Pitch = 1000",
            "Frequency = 10",
            "Volume (µl/cm) = 0.100000",
            "Type = Volume",
            "Field Data: ",
            "[0, 0, 0]",
        ]
    ),
    "2lagoon": "\r\n".join(
        [
            "Comment: ",
            "Field(s): ",
            "X = 1",
            "Y = 2",
            "Start Point",
            "Left: 29000",
            "Up: 2400",
            "X Field Gap:-20625/",
            "Y Field Gap:-17242/-19792/",
            "Pattern Size: ",
            "X = 56",
            "Y = 16",
            "Dot Pitch: ",
            "X = 375",
            "Y = 1212",
            "Type: No. of Drops",
            "Line Spotting:No",
            "Direction = Y",
            "Spot Pitch = 1000",
            "Frequency = 10",
            "Volume (µl/cm) = 0.100000",
            "Type = Volume",
            "Field Data: ",
            "[0, 0, 0]",
        ]
    ),
}

FOOTER = (
    "\r\n§07-12-38-51-20-12-22-14-55-52-57-29-27-20-36-66"
    "-49-49-67-56-08-58-57-62-46-64-38-43-41-11-16-27-63"
    "-11-45-13-08-51-38-40-19-23-39-64-56-26-44-38-32-68\r\n"
)
