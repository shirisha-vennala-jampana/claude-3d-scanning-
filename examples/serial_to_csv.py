"""Record the sensor logger's serial output to a CSV file.

    pip install pyserial
    python examples/serial_to_csv.py /dev/ttyUSB0 sensors.csv   # COM3 on Windows

Start this, then start filming and note the logger time shown when you press
record (it is printed every 10 s); pass it as --offset with a minus sign:
    scan3d sensor-log my_scene sensors.csv --offset -12.5
"""
import sys
import time

import serial  # pyserial


def main(port: str, out: str, baud: int = 115200) -> None:
    with serial.Serial(port, baud, timeout=1) as ser, open(out, "w") as f:
        f.write("timestamp,key,value,unit\n")
        last_print = 0.0
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line or line.startswith("timestamp") or line.count(",") != 3:
                continue
            f.write(line + "\n")
            f.flush()
            t = float(line.split(",")[0])
            if time.time() - last_print > 10:
                print(f"logger time {t:.1f} s  |  {line}")
                last_print = time.time()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
