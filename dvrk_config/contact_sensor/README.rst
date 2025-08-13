*****************
CONTACT SENSOR
*****************

HARDWARE
########
todo

SOFTWARE
########

ARDUINO
*******
Download the library `CapacitiveSensor <https://playground.arduino.cc/Main/CapacitiveSensor/index-2.html>`_.

Create as much instaces of the libraries as the number of your sensors. 

Use the ``capacitiveSensor(byte samples)`` method to retreive the value of the sensed capacitance of the object you are measuring.

	NOTE: this method id diffrent from ``capacitiveSensorRaw(byte samples)`` which returns the total capacitance (both sensed and unsensed). 

The samples parameter can be used to increase the returned resolution, at the expense of slower performance. The returned value is not averaged over the number of samples, and the total value is reported.

Look at your output values in the final setup and look for a reasonable value of the variable threshold.

You will have your binary signal on the selected ``OUTPUT_PIN``.


CONFIGURATION FILES
*******************

To configure the IOs of the dVRK for this sensor you need to create and XML file and iclude it the console (system) JSON file you are using to run the dVRK.

XML file 
=========

You need to include one line per digital input, so one line per sensor. The line should look like this:

.. code-block:: xml

	<DigitalIn BitID="2" BoardID="0" Debounce="0.001" Name="PSM1_contact" Pressed="0" Trigger="all" />

* **BitID:** it depends on the DOF port you are using. More details in `dMIB IOs reference <https://dvrk.readthedocs.io/main/pages/configuration/dmib-io.html>`_.

* **BoardID:** it depends on the dvrk controller you are using (PSM, ECM, MTM) and on the DOF port. See `Board ID <https://dvrk.readthedocs.io/main/pages/configuration/identifiers/board-id.html>`_.

* **Debounce:** expressend in seconds (amount of time below which a signal is ignored).
* **Name:** this will be the name of the ROS topic that publishes the state contact/non contact.

	NOTE: the rostopic is an *event* so it will publish every time there is a change of state. You can look at the the statistic of this type of rostopic with ``/stats/events/period_statistics``.

* **Pressed:** 0 --> active state = HIGH; 1 --> active state = LOW.
* **Trigger:** "all" considers both rise and fall; otherwise "rise" or "fall" (VERIFY!!!!).

JSON file
==========

You can modify an existing console (system) JSON file to include the reading of your digital input. See `IOs <https://dvrk.readthedocs.io/main/pages/configuration/system/io.html>`_.

In your JSON file you will have to add in the IOs section a new object: ``"configuration_files": ["name_of_your_file.xml"]``. You want to have something like this:

.. code-block:: json

	"IOs":
	   [
	     {
	       "name": "IO1",
	       "port": "fw",
	       "protocol": "broadcast-query-read-write",
	       "configuration_files": ["name_of_your_file.xml"]
	     }
	   ]



