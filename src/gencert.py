#!/usr/bin/env python3
"""Generate a self-signed TLS cert for the laptop offload server.

iOS Safari (and Android Chrome) only allow camera access over HTTPS, so the
laptop must serve over TLS. This makes a cert whose SANs cover every local IPv4
address + <LocalHostName>.local + localhost, so the phone can reach the laptop
by IP or name over a USB tether / hotspot.

RE-RUN THIS AFTER you start USB tethering, so the tether IP (e.g. 172.20.10.x)
is included in the cert. Then trust certs/cert.pem on the phone (see SETUP_DEMO.md).
"""
import os, re, socket, subprocess, datetime, ipaddress
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CERT_DIR = os.path.join(ROOT, "certs")
os.makedirs(CERT_DIR, exist_ok=True)


def local_ipv4s():
    ips = []
    try:
        out = subprocess.check_output(["ifconfig"], text=True)
        for m in re.finditer(r"inet (\d+\.\d+\.\d+\.\d+)", out):
            ip = m.group(1)
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips


def local_hostname():
    try:
        return subprocess.check_output(["scutil", "--get", "LocalHostName"], text=True).strip()
    except Exception:
        return socket.gethostname().split(".")[0]


ips = local_ipv4s()
host = local_hostname()
dns = ["localhost", f"{host}.local"]
san = ([x509.DNSName(d) for d in dns]
       + [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
       + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips])

key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"{host}.local")])
now = datetime.datetime.now(datetime.timezone.utc)
cert = (
    x509.CertificateBuilder()
    .subject_name(name).issuer_name(name)
    .public_key(key.public_key()).serial_number(x509.random_serial_number())
    .not_valid_before(now - datetime.timedelta(days=1))
    .not_valid_after(now + datetime.timedelta(days=120))
    .add_extension(x509.SubjectAlternativeName(san), critical=False)
    .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
    .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True,
                                 key_encipherment=True, content_commitment=False,
                                 data_encipherment=False, key_agreement=False,
                                 crl_sign=True, encipher_only=False, decipher_only=False),
                   critical=True)
    .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
    .sign(key, hashes.SHA256())
)
with open(os.path.join(CERT_DIR, "key.pem"), "wb") as f:
    f.write(key.private_bytes(serialization.Encoding.PEM,
                              serialization.PrivateFormat.TraditionalOpenSSL,
                              serialization.NoEncryption()))
with open(os.path.join(CERT_DIR, "cert.pem"), "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

print("wrote", os.path.join(CERT_DIR, "cert.pem"), "and key.pem")
print("SAN IPs:", ips or "(none — start tethering, then re-run)", " DNS:", dns)
print()
print("On the phone, after trusting cert.pem, open:")
for ip in ips:
    print(f"   https://{ip}:8443/")
print(f"   https://{host}.local:8443/")
