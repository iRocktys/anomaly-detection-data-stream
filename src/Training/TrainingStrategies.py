class TrainAllStrategy:
    def shouldTrain(self, trueLabel, isAttack):
        return True


class TrainOracleNormalStrategy:
    def shouldTrain(self, trueLabel, isAttack):
        return int(isAttack) == 0


class NoTrainingStrategy:
    def shouldTrain(self, trueLabel, isAttack):
        return False
