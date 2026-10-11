"""Visual identity: NN Group orange (taken from the NN logo) on calm black-and-white, Apple-like surfaces.

Every colour lives here; assets/style.css only reads the Mantine CSS variables this produces
(plus the logo gradient, defined once there as --dt-gradient)."""

ACCENT = "#EA650D"  # NN orange: send button, active item, links, citation pills

# Ten shades with ACCENT at index 6 (light mode) and the logo's brighter #EE7F00 at 5 (dark mode).
NN_ORANGE = [
    "#FFF5EC", "#FFE6D0", "#FFCB9E", "#FCAD6A", "#F7923F",
    "#EE7F00", ACCENT, "#E64415", "#B8360F", "#8A280B",
]

# Apple-style neutrals: light greys for surfaces, near-black for dark mode.
GRAY = [
    "#F5F5F7", "#EDEDF0", "#E3E3E8", "#D2D2D7", "#AEAEB2",
    "#8E8E93", "#6E6E73", "#48484A", "#3A3A3C", "#1D1D1F",
]
DARK = [
    "#F5F5F7", "#C7C7CC", "#A1A1A6", "#8E8E93", "#48484A",
    "#3A3A3C", "#2C2C2E", "#161618", "#111113", "#0B0B0C",
]

FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, "Segoe UI", Roboto, sans-serif'
FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"

THEME = {
    "primaryColor": "nn",
    "primaryShade": {"light": 6, "dark": 5},
    "colors": {"nn": NN_ORANGE, "gray": GRAY, "dark": DARK},
    "black": "#1A1A1A",
    "white": "#FFFFFF",
    "fontFamily": FONT,
    "headings": {"fontFamily": FONT, "fontWeight": "650"},
    "defaultRadius": "lg",
    "radius": {"xs": "6px", "sm": "8px", "md": "12px", "lg": "14px", "xl": "20px"},
    "shadows": {
        "xs": "0 1px 2px rgba(0,0,0,.04)",
        "sm": "0 1px 3px rgba(0,0,0,.05), 0 1px 2px rgba(0,0,0,.03)",
        "md": "0 4px 16px rgba(0,0,0,.06), 0 1px 3px rgba(0,0,0,.04)",
        "lg": "0 12px 40px rgba(0,0,0,.10), 0 2px 6px rgba(0,0,0,.04)",
    },
    "components": {
        "Button": {"defaultProps": {"radius": "xl", "fw": 500}},
        "ActionIcon": {"defaultProps": {"radius": "xl"}},
        "Badge": {"defaultProps": {"radius": "xl"}},
        "TextInput": {"defaultProps": {"radius": "md"}},
        "Paper": {"defaultProps": {"radius": "lg"}},
        "HoverCard": {"defaultProps": {"radius": "lg", "shadow": "lg"}},
        "Menu": {"defaultProps": {"radius": "lg", "shadow": "lg"}},
        "Popover": {"defaultProps": {"radius": "lg", "shadow": "lg"}},
    },
}
