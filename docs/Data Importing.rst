
Data Importing
==============

refers to data_importing.py

Classes for importing data into the roicat package.

Conventions:

* Data_roicat is the super class for all data objects.
  
* Data_roicat can be used to make a custom data object.
    * Subclasses like Data_suite2p and Data_caiman should be used
        to import data from files and convert it to a
        Data_roicat ingestable format.
    * Subclass import methods should be functions that return
        properly formatted data ready for the superclass to ingest.
    * Avoid directly setting attributes. Try to always use a
        .set_attribute() method.
    * Subclass import methods should operate at the multi-session
        level. That is, they should take in lists of objects
        corresponding to multiple sessions.
    * Subclasses should be able to initialize classification and
        tracking independently. Minimize interdependent attributes.

    * Users should have flexibility in the following switch-cases:
        - FOV_images:
            - From file
            - From object
            - Only specify FOV_height and FOV_width
  
    * Only default to importing from file if the file is deriving
        from a standardized format (ie suite2p or caiman). Do not
        require standardization for custom data objects like class
        labels.

Data_roicat
-----------

.. autoclass:: roicat.data_importing.Data_roicat
    :members:
    :show-inheritance:

Data_suite2p
------------

.. autoclass:: roicat.data_importing.Data_suite2p
    :members:
    :show-inheritance:

Data_caiman
-----------

.. autoclass:: roicat.data_importing.Data_caiman
    :members:
    :show-inheritance:
