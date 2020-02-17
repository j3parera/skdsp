import unittest
import numpy as np
from skdsp.util import buffer
from numpy.testing.utils import assert_equal


class UtilTest(unittest.TestCase):

    def test_buffer(self):
        with self.assertRaises(ValueError):
            buffer(np.r_[1:11], 4.1)

        with self.assertRaises(ValueError):
            buffer(np.r_[1:11], 4, 2, order='K')

        actual = buffer(np.r_[1:11], 4)
        expected = np.reshape(np.r_[1:11, [0.0]*2], [3, 4])
        assert_equal(actual, expected)

        actual = buffer(np.r_[1:11], 4, order='C')
        expected = np.reshape(np.r_[1:11, [0.0]*2], [3, 4])
        assert_equal(actual, expected)

        actual = buffer(np.r_[1:11], 4, order='F')
        expected = np.reshape(np.array(np.r_[1:11, [0.0]*2]), [3, 4]).T
        assert_equal(actual, expected)

        actual = buffer(np.r_[1:11], 4, 1)
        expected = np.reshape(np.r_[0:4, 3:7, 6:10, 9, 10, 0, 0], [4, 4])
        assert_equal(actual, expected)

        actual = buffer(np.r_[1:11], 4, 1, order='F')
        expected = np.reshape(np.r_[0:4, 3:7, 6:10, 9, 10, 0, 0], [4, 4]).T
        assert_equal(actual, expected)

        actual = buffer(np.r_[1:11], 4, 2)
        expected = np.reshape(np.r_[0, 0:3, 1:5, 3:7, 5:9, 7:11], [5, 4])
        assert_equal(actual, expected)

        actual = buffer(np.r_[1:11], 4, 2, order='F')
        expected = np.reshape(np.r_[0, 0:3, 1:5, 3:7, 5:9, 7:11], [5, 4]).T
        assert_equal(actual, expected)

        actual = buffer(np.r_[1:11], 4, 3)
        expected = np.c_[np.r_[0, 0, 0:8], np.r_[0, 0:9], np.r_[0:10],
                         np.r_[1:11]]
        assert_equal(actual, expected)

        with self.assertRaises(ValueError):
            buffer(np.r_[1:11], 4, 3.1)

        with self.assertRaises(ValueError):
            buffer(np.r_[1:11], 4, 4)

        actual = buffer(np.r_[1:11], 4, -1)
        expected = np.r_[1:5, 6:10].reshape([2, 4])
        assert_equal(actual, expected)

        actual = buffer(np.r_[1:11], 4, -2)
        expected = np.r_[1:5, 7:11].reshape([2, 4])
        assert_equal(actual, expected)

        actual = buffer(np.r_[1:11], 4, -3)
        expected = np.r_[1:5, 8:11, 0].reshape([2, 4])
        assert_equal(actual, expected)

        actual = buffer(np.r_[1:11], 4, 1, 11)
        expected = np.reshape(np.r_[11, 1:4, 3:7, 6:10, 9, 10, 0, 0], [4, 4])
        assert_equal(actual, expected)

        with self.assertRaises(ValueError):
            buffer(np.r_[1:11], 4, 1, [10, 11])

        actual = buffer(np.r_[1:11], 4, 1, 'nodelay')
        expected = np.reshape(np.r_[1:5, 4:8, 7:11], [3, 4])
        assert_equal(actual, expected)

        with self.assertRaises(ValueError):
            buffer(np.r_[1:11], 4, 1, 'badstring')

        actual = buffer(np.r_[1:11], 4, 2, 'nodelay')
        expected = np.reshape(np.r_[1:5, 3:7, 5:9, 7:11], [4, 4])
        assert_equal(actual, expected)

        actual = buffer(np.r_[1:11], 4, 3, [11, 12, 13])
        expected = np.c_[np.r_[11, 12, 13, 1:8], np.r_[12, 13, 1:9],
                         np.r_[13, 1:10], np.r_[1:11]]
        assert_equal(actual, expected)

        actual = buffer(np.r_[1:11], 4, 3, 'nodelay')
        expected = np.c_[np.r_[1:8], np.r_[2:9], np.r_[3:10], np.r_[4:11]]
        assert_equal(actual, expected)

        actual = buffer(np.r_[1:12], 4, -2, 1)
        expected = np.r_[2:6, 8:12].reshape([2, 4])
        assert_equal(actual, expected)

    # test
    #  [y, z] = buffer(1:12,4);
    #  assert (y, reshape(1:12,4,3));
    #  assert (z, zeros (1,0));

    # test
    #  [y, z] = buffer(1:11,4);
    #  assert (y, reshape(1:8,4,2));
    #  assert (z, [9, 10, 11]);

    # test
    #  [y, z] = buffer([1:12]',4);
    #  assert (y, reshape(1:12,4,3));
    #  assert (z, zeros (0,1));

    # test
    #  [y, z] = buffer([1:11]',4);
    #  assert (y, reshape(1:8,4,2));
    #  assert (z, [9; 10; 11]);

    # test
    #  [y,z,opt] = buffer(1:15,4,-2,1);
    #  assert (y, reshape([2:5,8:11],4,2));
    #  assert (z, [14, 15]);
    #  assert (opt, 0);

    # test
    #  [y,z,opt] = buffer(1:11,4,-2,1);
    #  assert (y, reshape([2:5,8:11],4,2));
    #  assert (z, zeros (1,0));
    #  assert (opt, 2);

    # test
    #  [y,z,opt] = buffer([1:15]',4,-2,1);
    #  assert (y, reshape([2:5,8:11],4,2));
    #  assert (z, [14; 15]);
    #  assert (opt, 0);

    # test
    #  [y,z,opt] = buffer([1:11]',4,-2,1);
    #  assert (y, reshape([2:5,8:11],4,2));
    #  assert (z, zeros (0, 1));
    #  assert (opt, 2);

    # test
    #  [y,z,opt] = buffer([1:11],5,2,[-1,0]);
    #  assert (y, reshape ([-1:3,2:6,5:9],[5,3]));
    #  assert (z, [10, 11]);
    #  assert (opt, [8; 9]);

    # test
    #  [y,z,opt] = buffer([1:11]',5,2,[-1,0]);
    #  assert (y, reshape ([-1:3,2:6,5:9],[5,3]));
    #  assert (z, [10; 11]);
    #  assert (opt, [8; 9]);

if __name__ == "__main__":
    unittest.main()
