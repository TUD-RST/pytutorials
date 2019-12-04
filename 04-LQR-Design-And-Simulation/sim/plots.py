import matplotlib.pyplot as plt
import pickle

fig = plt.figure(figsize=(6, 6))
lti0 = pickle.load(open("ludyk_lti_0.p", "rb"))
lti5 = pickle.load(open("ludyk_lti_5.p", "rb"))

ax1 = plt.subplot(221)
plt.plot(lti0["t"], lti0["x"][:, 0])
plt.plot(lti0["t"], lti0["x"][:, 1])
plt.plot(lti0["t"], lti0["xd"][:, 0], ls='--')
plt.plot(lti0["t"], lti0["xd"][:, 1], ls='--')
plt.title("Linearized at $t^*=0$")

ax3 = plt.subplot(223, sharex=ax1)
plt.plot(lti0["t"], lti0["u"][:])
plt.plot(lti0["t"], lti0["ud"][:], ls='--')
plt.xlabel("$t$")

ax2 = plt.subplot(222)
plt.plot(lti5["t"], lti5["x"][:, 0], label="$x_1$")
plt.plot(lti5["t"], lti5["x"][:, 1], label="$x_2$")
plt.plot(lti5["t"], lti5["xd"][:, 0], ls='--', label="$x^*_1$")
plt.plot(lti5["t"], lti5["xd"][:, 1], ls='--', label="$x^*_2$")
plt.legend()
plt.title("Linearized at $t^*=5$")

ax4 = plt.subplot(224, sharex=ax2)
plt.plot(lti5["t"], lti5["u"][:], label="$u$")
plt.plot(lti5["t"], lti5["ud"][:], ls='--', label="$u^*$")
plt.xlabel("$t$")
plt.legend()

plt.tight_layout()

plt.savefig("../doc/img/fig1.pdf")

fig = plt.figure(figsize=(6, 8))
pseudoltv = pickle.load(open("ludyk_pseudoltv.p", "rb"))

plt.subplot(311)
plt.plot(pseudoltv["t"], pseudoltv["x"][:, 0], label="$x_1$")
plt.plot(pseudoltv["t"], pseudoltv["x"][:, 1], label="$x_2$")
plt.plot(pseudoltv["t"], pseudoltv["xd"][:, 0], ls='--', label="$x^*_1$")
plt.plot(pseudoltv["t"], pseudoltv["xd"][:, 1], ls='--', label="$x^*_2$")
plt.legend()

plt.subplot(312)
plt.plot(pseudoltv["t"], pseudoltv["u"][:], label="$u$")
plt.plot(pseudoltv["t"], pseudoltv["ud"][:], ls='--', label="$u^*$")
plt.legend()

plt.subplot(313)
plt.plot(pseudoltv["t"], pseudoltv["K"][:, 0, 0], label="$k_1$")
plt.plot(pseudoltv["t"], pseudoltv["K"][:, 0, 1], label="$k_2$")
plt.legend()
plt.xlabel("$t$")

plt.savefig("../doc/img/fig2.pdf")

fig = plt.figure(figsize=(6, 8))
ltv = pickle.load(open("ludyk_ltv.p", "rb"))

plt.subplot(311)
plt.plot(ltv["t"], ltv["x"][:, 0], label="$x_1$")
plt.plot(ltv["t"], ltv["x"][:, 1], label="$x_2$")
plt.plot(ltv["t"], ltv["xd"][:, 0], ls='--', label="$x^*_1$")
plt.plot(ltv["t"], ltv["xd"][:, 1], ls='--', label="$x^*_2$")
plt.legend()

plt.subplot(312)
plt.plot(ltv["t"], ltv["u"][:], label="$u$")
plt.plot(ltv["t"], ltv["ud"][:], ls='--', label="$u^*$")
plt.legend()

plt.subplot(313)
plt.plot(ltv["t"], ltv["K"][:, 0, 0], label="$k_1$")
plt.plot(ltv["t"], ltv["K"][:, 0, 1], label="$k_2$")
plt.legend()
plt.xlabel("$t$")

plt.savefig("../doc/img/fig3.pdf")
plt.show()