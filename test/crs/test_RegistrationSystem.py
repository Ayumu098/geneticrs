from ...crs.RegistrationSystem import RegistrationSystem
from ...data.ScheduleDataset import ScheduleDataset
import torch


def test_complete():
    """Run the RegistrationSystem with a non-overlapping enlistment.
    """

    dataset = ScheduleDataset("test/crs/sample.csv")
    system = RegistrationSystem(dataset)

    complete_enlistment = torch.tensor(list(range(20)))

    # Multiple runs to account for random nature of enlistment
    for _ in range(1000):
        assert system(complete_enlistment).equal(complete_enlistment)


def test_empty():
    """Run the RegistrationSystem with an empty enlistment.
    """
    dataset = ScheduleDataset("test/crs/sample.csv")
    system = RegistrationSystem(dataset)

    empty_enlistment = torch.tensor([])
    assert system(empty_enlistment).equal(empty_enlistment)


def test_overlap():
    """Test the RegistrationSystem with an overlapping enlistment.
    """
    dataset = ScheduleDataset("test/crs/sample.csv")
    system = RegistrationSystem(dataset)

    empty_enlistment = torch.randint(23, 35, (20,))
    for _ in range(1000):
        assert not system(empty_enlistment).equal(empty_enlistment)

