class CryptographicOperations:
    """
    Base class for cryptographic operations with timing latencies.
    """
    def __init__(self, device_name):
        self.device_name = device_name

    def T_X(self):
        """ECC point multiplication"""
        raise NotImplementedError

    def T_add(self):
        """ECC point addition"""
        raise NotImplementedError

    def T_S(self):
        """Scalar multiplication"""
        raise NotImplementedError

    def T_H(self):
        """One-way hash"""
        raise NotImplementedError

    def T_E(self):
        """Encryption"""
        raise NotImplementedError

    def T_D(self):
        """Decryption"""
        raise NotImplementedError

    def T_FE(self):
        """Fuzzy extractor"""
        raise NotImplementedError

    def T_PF(self):
        """PUF function"""
        raise NotImplementedError

    def T_RN(self):
        """Extraction of random number"""
        raise NotImplementedError

    def T_XOR(self):
        """XOR operation"""
        raise NotImplementedError

    def T_concat(self):
        """Concatenation operation"""
        raise NotImplementedError

    def run_all_operations(self):
        """Run all operations and return a dictionary of timings in ms."""
        return {
            "T_X": self.T_X(),
            "T_add": self.T_add(),
            "T_S": self.T_S(),
            "T_H": self.T_H(),
            "T_E": self.T_E(),
            "T_D": self.T_D(),
            "T_FE": self.T_FE(),
            "T_PF": self.T_PF(),
            "T_RN": self.T_RN(),
            "T_XOR": self.T_XOR(),
            "T_concat": self.T_concat()
        }


class Laptop(CryptographicOperations):
    """Core i7 laptop (C2)"""
    def __init__(self):
        super().__init__("Laptop (C2)")

    def T_X(self):
        return 0.445

    def T_add(self):
        return 0.0018

    def T_S(self):
        return 0.0138

    def T_H(self):
        return 0.149

    def T_E(self):
        return 0.461

    def T_D(self):
        return 0.461

    def T_FE(self):
        return 0.574

    def T_PF(self):
        return 0.00054

    def T_RN(self):
        return 2.011

    def T_XOR(self):
        return 0.0000106

    def T_concat(self):
        return 0.0000733


class CellPhone(CryptographicOperations):
    """Samsung Galaxy A21s (OD)"""
    def __init__(self):
        super().__init__("CellPhone (OD)")

    def T_X(self):
        return 0.405

    def T_add(self):
        return 0.035

    def T_S(self):
        return 0.830

    def T_H(self):
        return 0.674

    def T_E(self):
        return 0.851

    def T_D(self):
        return 0.851

    def T_FE(self):
        return 2.288

    def T_PF(self):
        return 0.00054

    def T_RN(self):
        return 2.448

    def T_XOR(self):
        return 0.00000442

    def T_concat(self):
        return 0.00000854


class RaspberryPi(CryptographicOperations):
    """Raspberry Pi 5 (RQ)"""
    def __init__(self):
        super().__init__("Raspberry Pi (RQ)")

    def T_X(self):
        return 0.211

    def T_add(self):
        return 0.323

    def T_S(self):
        return 0.474

    def T_H(self):
        return 0.891

    def T_E(self):
        return 1.014

    def T_D(self):
        return 1.014

    def T_FE(self):
        return 4.076

    def T_PF(self):
        return 0.00054

    def T_RN(self):
        return 2.946

    def T_XOR(self):
        return 0.0000025

    def T_concat(self):
        return 0.0001007


# ------------------- Test Execution -------------------
if __name__ == "__main__":
    c2 = Laptop()
    od = CellPhone()
    rq = RaspberryPi()

    print("C2 (Laptop) Timings (ms):")
    for op, time in c2.run_all_operations().items():
        print(f"  {op}: {time}")

    print("\nOD (CellPhone) Timings (ms):")
    for op, time in od.run_all_operations().items():
        print(f"  {op}: {time}")

    print("\nRQ (Raspberry Pi) Timings (ms):")
    for op, time in rq.run_all_operations().items():
        print(f"  {op}: {time}")