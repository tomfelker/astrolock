import win32com.client

factory = win32com.client.Dispatch("LocationDisp.LatLongReportFactory")
factory.RequestPermissions(0)
factory.ListenForReports(1000)
