import PyPDF2
import wx
import custom_gui as cg
import re



class HomeFrame(wx.Frame):
    """GUI class manages opening EWave unit COMs"""
    ewaves = {}

    def __init__(self, win_title):
        wx.Frame.__init__(self, None, title=win_title, pos=(150, 150), size=(350, 200))
        icon = wx.EmptyIcon()
        icon.CopyFromBitmap(wx.Bitmap("res\grab.ico", wx.BITMAP_TYPE_ANY))
        self.SetIcon(icon)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Set Menu
        menuBar = wx.MenuBar()
        menu = wx.Menu()
        m_exit = menu.Append(wx.ID_EDIT, "&Exit\talt-x", "Close window")
        self.Bind(wx.EVT_MENU, self.on_close, m_exit)
        menuBar.Append(menu, "&File")
        menu = wx.Menu()
        m_about = menu.Append(wx.ID_ABOUT, "&About", "Information about this program")
        self.Bind(wx.EVT_MENU, cg.on_about, m_about)
        menuBar.Append(menu, "&Help")
        self.SetMenuBar(menuBar)

        self.all_ports = []
        self.xab_logo = ''

        # sets up listener for windows events outside of wx scope to intercept changes to USB hub.
        # required to auto-refresh list of available com ports
        self.oldWndProc = win32gui.SetWindowLong(
            self.GetHandle(),
            win32con.GWL_WNDPROC,
            self.wnd_proc_intercept)

        #  Make a dictionary of message names to be used for printing below
        self.msgdict = {}
        for name in dir(win32con):
            if name.startswith("WM_"):
                value = getattr(win32con, name)
                self.msgdict[value] = name

        # Set up widget layout
        self.__do_layout()

    def wnd_proc_intercept(self, h_wnd, msg, w_param, l_param):

        #  Check for device nodes change. This will catch attach and eject, w/o info
        if w_param == ewcommands.DBT_DEVNODES_CHANGED:
            self.update_com_list()

        # Restore the old WndProc.  Notice the use of win32api
        #  instead of win32gui here.  This is to avoid an error due to
        #  not passing a callable object.
        if msg == win32con.WM_DESTROY:
            win32api.SetWindowLong(self.GetHandle(),
                                   win32con.GWL_WNDPROC, self.oldWndProc)

        # Pass all messages (in this case, yours may be different) on
        #  to the original WndProc
        return win32gui.CallWindowProc(self.oldWndProc, h_wnd,
                                       msg, w_param, l_param)

    def __do_layout(self):
        top_panel = wx.Panel(self)
        right_panel = wx.Panel(top_panel)
        left_panel = wx.Panel(top_panel)
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        img_sizer = wx.BoxSizer(wx.VERTICAL)
        dropdown_sizer = wx.BoxSizer(wx.VERTICAL)
        serial_sizer = wx.BoxSizer(wx.VERTICAL)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        dropdown_label = wx.StaticText(right_panel, label="Select E'Wave Unit Address:", style=wx.ALIGN_LEFT)
        self.addresses = [str(i) for i in range(32)]
        self.address_dropdown = wx.ComboBox(right_panel, wx.ID_ANY, choices=self.addresses, value=self.addresses[0],
                                            style=wx.CB_READONLY | wx.CB_DROPDOWN | wx.ALIGN_LEFT, size=(50, 25))
        self.address_dropdown.SetSelection(ewcommands.DEFAULT_ADDRESS)

        com_select_lbl = wx.StaticText(right_panel, label="Select E'Wave Serial COM:", style=wx.CENTER | wx.ALL)
        self.com_list = wx.ListBox(right_panel, choices=["None"], name="Select E'Wave Serial COM:", size=(150, 100))

        self.launch_btn = wx.Button(right_panel, wx.ID_ANY, "Launch")
        self.exit_btn = wx.Button(right_panel, wx.ID_ANY, "Exit")
        self.Bind(wx.EVT_BUTTON, self.on_close, self.exit_btn)
        self.Bind(wx.EVT_BUTTON, self.open_unit_control_dialog, self.launch_btn)
        self.launch_btn.SetFocus()

        logo = wx.Image('res/aeLogo.png', wx.BITMAP_TYPE_ANY)
        width, height = (logo.GetWidth(), logo.GetHeight())
        scale = 100.0 / width
        logo = logo.Scale(width * scale, height * scale, wx.IMAGE_QUALITY_HIGH)
        logo = wx.BitmapFromImage(logo)
        ae_logo_bitmap = wx.StaticBitmap(left_panel, wx.ID_ANY, logo, (10, 5), (logo.GetWidth(), logo.GetHeight()))

        logo = wx.Image('res/xab_logo.png', wx.BITMAP_TYPE_ANY)
        width, height = (logo.GetWidth(), logo.GetHeight())
        scale = 150.0 / width
        logo = logo.Scale(width * scale, height * scale, wx.IMAGE_QUALITY_HIGH)
        logo = wx.BitmapFromImage(logo)
        self.xab_logo = logo
        xab_logo_bitmap = wx.StaticBitmap(left_panel, -1, logo, (10, 5), (logo.GetWidth(), logo.GetHeight()))

        # RIGHT PANEL
        dropdown_sizer.Add(dropdown_label, flag=wx.ALL | wx.CENTER, border=5)
        dropdown_sizer.Add(self.address_dropdown,flag=wx.ALL | wx.CENTER, border=5)

        serial_sizer.Add(com_select_lbl, flag=wx.ALL | wx.CENTER, border=10)
        serial_sizer.Add(self.com_list, flag=wx.ALL | wx.CENTER, border=10)

        btn_sizer.Add(self.launch_btn, flag=wx.ALL, border=5)
        btn_sizer.Add(self.exit_btn, flag=wx.ALL, border=5)

        right_sizer.Add(dropdown_sizer, 0, wx.ALL | wx.EXPAND, 10)
        right_sizer.Add(serial_sizer, 0, wx.ALL | wx.EXPAND, 10)
        right_sizer.AddStretchSpacer(prop=1)
        right_sizer.Add(btn_sizer, 0, wx.ALL | wx.EXPAND, 10)
        right_panel.SetSizer(right_sizer)

        # LEFT PANEL
        img_sizer.Add(xab_logo_bitmap, 0, wx.TOP | wx.CENTER | wx.EXPAND, 20)
        img_sizer.AddStretchSpacer(prop=1)
        img_sizer.Add(ae_logo_bitmap, 0, wx.BOTTOM | wx.CENTER | wx.EXPAND, 20)
        left_panel.SetSizer(img_sizer)

        # TOP PANEL
        top_sizer.Add(left_panel, 0, wx.ALL | wx.EXPAND, 10)
        top_sizer.Add(wx.StaticLine(top_panel, wx.ID_ANY, style=wx.LI_VERTICAL), 0, wx.ALL | wx.EXPAND)
        top_sizer.AddStretchSpacer(prop=1)
        top_sizer.Add(right_panel, 0, wx.ALL | wx.EXPAND, 10)

        top_panel.SetSizer(top_sizer)
        top_sizer.Fit(self)
        self.update_com_list()

    def enumerate_serial_ports(self):
        """ Uses the Win32 registry to return an
            iterator of serial (COM) ports
            existing on this computer.
        """
        self.all_ports = []
        available_ports = list_ports.comports()
        for device in available_ports:
            self.all_ports.append(device)
        return self.all_ports

    def update_com_list(self):
        choices = self.enumerate_serial_ports()
        if not choices:
            choices = [["None", "No Devices Detected", "0x0000"]]
        self.com_list.Clear()
        for ind, name, val in choices:
            self.com_list.Append(name)
        self.com_list.SetSelection(0)

    def open_unit_control_dialog(self, event=None):
        new_address = self.address_dropdown.GetValue()
        if new_address in [unit.address for unit in ewave.Ewave.get_instances()]:
            cg.dlg_message(self, "Control for unit " + str(new_address) + " already open!", "Already Open!")
            return

        if self.com_list.GetSelection() is not "NOT_FOUND":
            try:
                port_selected = int([re.findall(r'\d+',
                                                self.com_list.GetString(self.com_list.GetSelection()))][0][0])
            except IndexError:
                event.Skip()
                cg.dlg_message(self,
                               "Something went wrong trying to get that com port. Tell a friend or try again.",
                               "Invalid Selection")
                return
        else:
            event.Skip()
            cg.dlg_message(self, "Select a valid port from the list or check USB connections", "Invalid Selection")
            return

        port_name = self.com_list.GetString(self.com_list.GetSelection())
        port_number = port_selected - 1  # something about zero indexing, windows adds one to COM port numbers.
        board_num = 0
        if use_device_detection:
            ul.ignore_instacal()
            if not util.config_first_detected_device(board_num):
                cg.dlg_message(self, "Couldn't find an MCC DAQ.  Check USB.", "No DAQ!", icon=wx.ICON_ERROR)
                event.Skip()
                return
        else:
            # TODO handle case where board config is loaded automatically from CB.CFG, see MCC UL docs for deets
            event.Skip()
            return
        try:
            serial_ver = float(serial.VERSION)
        except ValueError:
            serial_ver = 0
		# cg.dlg_message(self, port_number, "")
        if serial_ver >= 3 and serial_ver > 0:
            _port = "COM" + str(port_number + 1)
        else:
            _port = port_number
        ewave_control = ewave.Ewave(com=_port, address=new_address)
        ewave_control.Show()

    def on_close(self, event=None):
        for ewave_window in ewave.Ewave.get_instances():
            ewave_window.on_close()
        self.Destroy()


def run_main():
    app = wx.App(redirect=False)
    app.frame = HomeFrame("E'Wave Control")
    app.frame.Show()
    app.SetTopWindow(app.frame)
    app.MainLoop()

if __name__ == '__main__':
    run_main()
