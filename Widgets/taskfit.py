from qt_task import Task
# from multiprocessing import Process


class TaskFit(Task):
    def __init__(self, fw, parent=None):
        super(TaskFit, self).__init__(parent)
        self.fw = fw  # fit widget

    def preRun(self):
        self.fw.set_btns_enabled(False)  # disable buttons
        self.fw.fitter.is_interruption_requested = self.isInterruptionRequested  # setup a function

    def run(self):
        if self.fw.cbVerbose.isChecked():
            print('Fitting...')
        self.run_fit()

    def run_fit(self):
        # hard model fit
        if self.fw.current_model.connectivity.count(0) == 0 or self.fw.current_model._class == 'Femto':
            if self.fw.current_model._class == 'Femto':
                self.fw.D_fit = self.fw.fitter.var_pro_femto()
            elif self.fw.current_model._class == 'Steady state photokinetics':
                if self.fw.current_model.method == 'RFA':
                    self.fw.fitter.obj_func_fit()  # TODO>>>
                elif self.fw.current_model.method == 'HS-MCR-ALS':
                    self.fw.fitter.HS_MCR_fit(c_model=self.fw.current_model)
                else:  # varible projection
                    self.fw.fitter.fit_full_model()
            else:
                self.fw.fitter.fit_full_model()

        elif self.fw.current_model.connectivity.count( 0) == int(self.fw.sbN.value()):  # pure MCR fit
            self.fw.fitter.HS_MCR_fit(c_model=None)
        else:  # mix of two, HS-fit
            self.fw.fitter.HS_MCR_fit(c_model=self.fw.current_model)

    def postRun(self):  # after run has finished
        if self.fw.fitter.c_model is not None:
            self.fw.current_model = self.fw.fitter.c_model

        self.fw.update_model_par_count(update_after_fit=True)

        self.fw._C = self.fw.fitter.C_opt
        self.fw._ST = self.fw.fitter.ST_opt

        self.fw.plot_opt_matrices()
        if self.fw.cbVerbose.isChecked():
            self.fw.print_stats()

        self.fw.set_btns_enabled(True)  # enable buttons

