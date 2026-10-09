// Runs before the app renders, so the page never flashes the wrong theme.
import { applyTheme, storedTheme } from "./lib/theme";

applyTheme(storedTheme());
