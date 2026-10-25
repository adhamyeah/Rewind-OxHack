"""Optional potentiometer on a serial port: the Arduino prints 0..1023 lines (see knob/knob.ino)."""

import threading


class Knob:
    def __init__(self, port, baud=9600):
        self.value = None   # latest reading 0..1023, or None until one arrives
        self.ok = False
        try:
            import serial

            self.ser = serial.Serial(port, baud, timeout=1)
            self.ok = True
        except Exception as e:
            print(f"[knob] unavailable, running without it: {e}")
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line.isdigit():
                    self.value = max(0, min(1023, int(line)))
            except Exception as e:
                print(f"[knob] lost: {e}")
                self.ok = False
                return

    def position(self, n):
        """Map the reading to 0..n (n = live) or None if no reading yet."""
        if self.value is None:
            return None
        return round(self.value / 1023 * n)

    def stop(self):
        self.running = False
