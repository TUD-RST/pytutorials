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
plt.ylabel("state")
plt.title("Linearized at $t^*=0$")

ax3 = plt.subplot(223, sharex=ax1)
plt.plot(lti0["t"], lti0["u"][:])
plt.plot(lti0["t"], lti0["ud"][:], ls='--')
plt.xlabel("$t$")
plt.ylabel("input")

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

plt.savefig("../doc/img/ludyk_lti.pdf")

fig = plt.figure(figsize=(6, 8))
pseudoltv = pickle.load(open("ludyk_pseudoltv.p", "rb"))

plt.subplot(311)
plt.plot(pseudoltv["t"], pseudoltv["x"][:, 0], label="$x_1$")
plt.plot(pseudoltv["t"], pseudoltv["x"][:, 1], label="$x_2$")
plt.plot(pseudoltv["t"], pseudoltv["xd"][:, 0], ls='--', label="$x^*_1$")
plt.plot(pseudoltv["t"], pseudoltv["xd"][:, 1], ls='--', label="$x^*_2$")
plt.ylabel("state")
plt.legend()

plt.subplot(312)
plt.plot(pseudoltv["t"], pseudoltv["u"][:], label="$u$")
plt.plot(pseudoltv["t"], pseudoltv["ud"][:], ls='--', label="$u^*$")
plt.ylabel("input")
plt.legend()

plt.subplot(313)
plt.plot(pseudoltv["t"], pseudoltv["K"][:, 0, 0], label="$k_1$")
plt.plot(pseudoltv["t"], pseudoltv["K"][:, 0, 1], label="$k_2$")
plt.xlabel("$t$")
plt.ylabel("feedback gain")
plt.legend()

plt.savefig("../doc/img/ludyk_pseudoltv.pdf")

fig = plt.figure(figsize=(6, 8))
ltv = pickle.load(open("ludyk_ltv.p", "rb"))

plt.subplot(311)
plt.plot(ltv["t"], ltv["x"][:, 0], label="$x_1$")
plt.plot(ltv["t"], ltv["x"][:, 1], label="$x_2$")
plt.plot(ltv["t"], ltv["xd"][:, 0], ls='--', label="$x^*_1$")
plt.plot(ltv["t"], ltv["xd"][:, 1], ls='--', label="$x^*_2$")
plt.ylabel("state")
plt.legend()

plt.subplot(312)
plt.plot(ltv["t"], ltv["u"][:], label="$u$")
plt.plot(ltv["t"], ltv["ud"][:], ls='--', label="$u^*$")
plt.ylabel("input")
plt.legend()

plt.subplot(313)
plt.plot(ltv["t"], ltv["K"][:, 0, 0], label="$k_1$")
plt.plot(ltv["t"], ltv["K"][:, 0, 1], label="$k_2$")
plt.xlabel("$t$")
plt.ylabel("feedback gain")
plt.legend()

plt.savefig("../doc/img/ludyk_ltv.pdf")

fig = plt.figure(figsize=(6, 8))
pendulum_lti = pickle.load(open("pendulum_lti.p", "rb"))

plt.subplot(311)
plt.plot(pendulum_lti["t"], pendulum_lti["x"][:, 0], label="$s$")
plt.plot(pendulum_lti["t"], pendulum_lti["x"][:, 2], label="$\\varphi$")
plt.plot(pendulum_lti["t"], pendulum_lti["xd"][:, 0], ls='--', label="$s^*$")
plt.plot(pendulum_lti["t"], pendulum_lti["xd"][:, 2], ls='--', label="$\\varphi^*$")
plt.ylim(-2, 5)
plt.ylabel("state")
plt.legend()

plt.subplot(312)
plt.plot(pendulum_lti["t"], pendulum_lti["u"][:], label="$u$")
plt.plot(pendulum_lti["t"], pendulum_lti["ud"][:], ls='--', label="$u^*$")
plt.ylabel("input")
plt.ylim(-15, 10)
plt.legend()

plt.subplot(313)
plt.plot(pendulum_lti["t"], pendulum_lti["K"][:, 0, 0])
plt.plot(pendulum_lti["t"], pendulum_lti["K"][:, 0, 1])
plt.plot(pendulum_lti["t"], pendulum_lti["K"][:, 0, 2])
plt.plot(pendulum_lti["t"], pendulum_lti["K"][:, 0, 3])
plt.xlabel("$t$")
plt.ylabel("feedback gain")

plt.savefig("../doc/img/pendulum_lti.pdf")

fig = plt.figure(figsize=(6, 8))
pendulum_pseudoltv = pickle.load(open("pendulum_pseudoltv.p", "rb"))

plt.subplot(311)
plt.plot(pendulum_pseudoltv["t"], pendulum_pseudoltv["x"][:, 0], label="$s$")
plt.plot(pendulum_pseudoltv["t"], pendulum_pseudoltv["x"][:, 2], label="$\\varphi$")
plt.plot(pendulum_pseudoltv["t"], pendulum_pseudoltv["xd"][:, 0], ls='--', label="$s^*$")
plt.plot(pendulum_pseudoltv["t"], pendulum_pseudoltv["xd"][:, 2], ls='--', label="$\\varphi^*$")
plt.ylim(-2, 5)
plt.ylabel("state")
plt.legend()

plt.subplot(312)
plt.plot(pendulum_pseudoltv["t"], pendulum_pseudoltv["u"][:], label="$u$")
plt.plot(pendulum_pseudoltv["t"], pendulum_pseudoltv["ud"][:], ls='--', label="$u^*$")
plt.ylabel("input")
plt.ylim(-15, 10)
plt.legend()

plt.subplot(313)
plt.plot(pendulum_pseudoltv["t"], pendulum_pseudoltv["K"][:, 0, 0])
plt.plot(pendulum_pseudoltv["t"], pendulum_pseudoltv["K"][:, 0, 1])
plt.plot(pendulum_pseudoltv["t"], pendulum_pseudoltv["K"][:, 0, 2])
plt.plot(pendulum_pseudoltv["t"], pendulum_pseudoltv["K"][:, 0, 3])
plt.xlabel("$t$")
plt.ylabel("feedback gain")

plt.savefig("../doc/img/pendulum_pseudoltv.pdf")

fig = plt.figure(figsize=(6, 8))
pendulum_ltv = pickle.load(open("pendulum_ltv.p", "rb"))

plt.subplot(311)
plt.plot(pendulum_ltv["t"], pendulum_ltv["x"][:, 0], label="$s$")
plt.plot(pendulum_ltv["t"], pendulum_ltv["x"][:, 2], label="$\\varphi$")
plt.plot(pendulum_ltv["t"], pendulum_ltv["xd"][:, 0], ls='--', label="$s^*$")
plt.plot(pendulum_ltv["t"], pendulum_ltv["xd"][:, 2], ls='--', label="$\\varphi^*$")
plt.ylim(-2, 5)
plt.ylabel("state")
plt.legend()

plt.subplot(312)
plt.plot(pendulum_ltv["t"], pendulum_ltv["u"][:], label="$u$")
plt.plot(pendulum_ltv["t"], pendulum_ltv["ud"][:], ls='--', label="$u^*$")
plt.ylabel("input")
plt.ylim(-15, 10)
plt.legend()

plt.subplot(313)
plt.plot(pendulum_ltv["t"], pendulum_ltv["K"][:, 0, 0])
plt.plot(pendulum_ltv["t"], pendulum_ltv["K"][:, 0, 1])
plt.plot(pendulum_ltv["t"], pendulum_ltv["K"][:, 0, 2])
plt.plot(pendulum_ltv["t"], pendulum_ltv["K"][:, 0, 3])
plt.xlabel("$t$")
plt.ylabel("feedback gain")

plt.savefig("../doc/img/pendulum_ltv.pdf")

plt.show()