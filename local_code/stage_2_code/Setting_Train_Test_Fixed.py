'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting


class Setting_Train_Test_Fixed(setting):
    def load_run_save_evaluate(self):
        # load dataset that already contains train/test sets
        loaded_data = self.dataset.load()

        # pass the pre-split data directly into the method
        self.method.data = {
            'train': loaded_data['train'],
            'test': loaded_data['test']
        }

        # run the learning method
        learned_result = self.method.run()

        # save raw results
        self.result.data = learned_result
        self.result.fold_count = 0
        self.result.save()

        # evaluate the result
        self.evaluate.data = learned_result
        return self.evaluate.evaluate(), None

        