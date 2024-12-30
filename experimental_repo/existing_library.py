from mpyc.runtime import mpc


async def main():
    secint = mpc.SecInt(16)

    await mpc.start()

    my_age = int(input("Enter your age: "))
    our_ages = mpc.input(secint(my_age))

    max_age = mpc.min(our_ages)

    print("Minimum age:", await mpc.output(max_age))

    await mpc.shutdown()


mpc.run(main())
