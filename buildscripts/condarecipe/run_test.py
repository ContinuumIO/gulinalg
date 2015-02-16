import sys
import gulinalg

sys.argv += '-m -b'.split()

if not gulinalg.test():
    print("Test failed")
    sys.exit(1)

print('gulinalg.__version__: {0}'.format(gulinalg.__version__))
