from pathlib import Path
import unittest


SOURCE = Path(__file__).with_name("main.cpp").read_text(encoding="utf-8")


def function_body(signature: str, next_signature: str) -> str:
    return SOURCE.split(signature, 1)[1].split(next_signature, 1)[0]


class I2cScanSourceTest(unittest.TestCase):
    def test_live_firmware_prints_gpio_driver_version(self) -> None:
        self.assertIn('serial_print("Firmware: GPIO-I2C v2\\n");', SOURCE)

    def test_unused_usci_transaction_driver_is_not_kept_in_flash(self) -> None:
        self.assertNotIn("static void i2c_init(void)", SOURCE)
        self.assertNotIn("static int i2c_write_reg(", SOURCE)
        self.assertNotIn("static int i2c_read_bytes(", SOURCE)

    def test_scan_does_not_leave_usci_in_a_started_transaction(self) -> None:
        scan = function_body(
            "static void i2c_scan(void) {",
            "static int mpu6050_init_dev(void) {",
        )

        self.assertNotIn("UCB0CTL1 |= UCTR | UCTXSTT;", scan)
        self.assertIn("sw_i2c_init();", scan)

    def test_mpu6050_uses_the_working_software_i2c_path(self) -> None:
        init = function_body(
            "static int mpu6050_init_dev(void) {",
            "static int mpu6050_read_accel(",
        )
        read = function_body(
            "static int mpu6050_read_accel(",
            "// ============================================================================\n"
            "// LIVE MODE",
        )

        self.assertIn("sw_i2c_write_reg(", init)
        self.assertIn("sw_i2c_read_bytes(", read)

    def test_software_i2c_rejects_a_stuck_low_bus(self) -> None:
        ping = function_body(
            "static int sw_ping_mpu(void) {",
            "static uint8_t sw_i2c_read_byte(",
        )
        write = function_body(
            "static int sw_i2c_write_reg(",
            "static int sw_i2c_read_bytes(",
        )

        self.assertIn("if (!sw_i2c_bus_idle()) return 0;", ping)
        self.assertIn("if (!sw_i2c_bus_idle()) return -1;", write)


if __name__ == "__main__":
    unittest.main()
