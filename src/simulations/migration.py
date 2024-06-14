
from vice.toolkit import hydrodisk
import random
import math as m
#from vice._globals import END_TIME, ZONE_WIDTH
from .._globals import ZONE_WIDTH, END_TIME

class diskmigration(hydrodisk.hydrodiskstars):

	r"""
	A ``hydrodiskstars`` object which writes extra analog star particle data to
	an output file.

	Parameters
	----------
	radbins : array-like
		The bins in galactocentric radius in kpc corresponding to each annulus.
	mode : ``str`` [default : "linear"]
		A keyword denoting the time-dependence of stellar migration.
		Allowed values:

		- "diffusion"
		- "linear"
		- "sudden"
		- "post-process"

	filename : ``str`` [default : "stars.out"]
		The name of the file to write the extra star particle data to.

	Attributes
	----------
	write : ``bool`` [default : False]
		A boolean describing whether or not to write to an output file when
		the object is called. The ``multizone`` object, and by extension the
		``milkyway`` object, automatically switch this attribute to True at the
		correct time to record extra data.
	"""

	def __init__(self, radbins, mode = "diffusion", filename = "stars.out",
		**kwargs):
		super().__init__(radbins, mode = mode, **kwargs)
		if isinstance(filename, str):
			self._file = open(filename, 'w')
			self._file.write("# zone_origin\ttime_origin\tanalog_id\tzfinal\n")
		else:
			raise TypeError("Filename must be a string. Got: %s" % (
				type(filename)))

		# use only disk stars in these simulations
		self.decomp_filter([1, 2])

		# Multizone object automatically swaps this to True in setting up
		# its stellar population zone histories
		self.write = False

	def __call__(self, zone, tform, time):
		if tform == time:
			super().__call__(zone, tform, time) # reset analog star particle
			if self.write:
				if self.analog_index == -1:
					# finalz = 100
					finalz = 0
					analog_id = -1
				else:
					finalz = self.analog_data["zfinal"][self.analog_index]
					analog_id = self.analog_data["id"][self.analog_index]
				self._file.write("%d\t%.2f\t%d\t%.2f\n" % (zone, tform,
					analog_id, finalz))
			else: pass
			return zone
		else:
			return super().__call__(zone, tform, time)

	def close_file(self):
		r"""
		Closes the output file - should be called after the multizone model
		simulation runs.
		"""
		self._file.close()

	@property
	def write(self):
		r"""
		Type : bool

		Whether or not to write out to the extra star particle data output
		file. For internal use by the vice.multizone object only.
		"""
		return self._write

	@write.setter
	def write(self, value):
		if isinstance(value, bool):
			self._write = value
		else:
			raise TypeError("Must be a boolean. Got: %s" % (type(value)))


class gaussian_migration:
	r"""
	A class which controls the Gaussian stellar migration scheme.

	The total migration distance $\Delta R$ is drawn from a Gaussian whose
	width is determined by the star's age and birth radius. The final z-height
	is drawn from a sech^2 distribution and written to an external file.

	Parameters
	----------
	radbins : array-like
		The bins in galactocentric radius in kpc corresponding to each annulus.
	zone_width : float [default : 0.1]
		Width of each radial zone in kpc.
	end_time : float [default : 13.2]
		The final simulation timestep in Gyr.
	filename : ``str`` [default : "stars.out"]
		The name of the file to write the extra star particle data to.
	absz_max : float [default : 3]
		Maximum |z|-height above the midplane in kpc. The default corresponds
		to the maximum value in the h277 sample.

	Attributes
	----------
	write : ``bool`` [default : False]
		A boolean describing whether or not to write to an output file when
		the object is called. The ``multizone`` object, and by extension the
		``milkyway`` object, automatically switch this attribute to True at the
		correct time to record extra data.

	Calling
	-------
	Returns the star's current zone at the given time.
	zone : int
		Birth zone of the star.
	tform : float
		Formation time of the star in Gyr.
	time : float
		Current simulation time in Gyr.
	"""
	def __init__(self, radbins, zone_width = ZONE_WIDTH, end_time = END_TIME,
			filename = "stars.out", absz_max = 3., post_process = False):
		self.radial_bins = radbins
		self.zone_width = zone_width
		self.end_time = end_time
		self.absz_max = absz_max
		self.post_process = post_process
		# super().__init__(radbins, mode=None, filename=filename, **kwargs)
		if isinstance(filename, str):
			self._file = open(filename, 'w')
			# Same format as above for compatibility
			self._file.write("# zone_origin\ttime_origin\tanalog_id\tzfinal\n")
		else:
			raise TypeError("Filename must be a string. Got: %s" % (
				type(filename)))

		# Multizone object automatically swaps this to True in setting up
		# its stellar population zone histories
		self.write = False

	def __call__(self, zone, tform, time):
		Rform = self.zone_width * (zone + 0.5)
		age = self.end_time - tform
		if tform == time:
			if age > 0:
				# Randomly draw migration distance dR based on age & Rform
				while True: # ensure 0 < Rfinal <= Rmax
					dR = random.gauss(mu=0., sigma=self.migr_scale(age, Rform))
					Rfinal = Rform + dR
					if Rfinal > 0. and Rfinal <= self.radial_bins[-1]:
						break
			else:
				# Stars born in the last simulation timestep won't migrate
				dR = 0.
				Rfinal = Rform
			self.dR = dR
			# Randomly draw final midplane distance and write to file
			if self.write:
				# vertical scale height based on age and Rfinal
				hz = self.scale_height(age, Rfinal)
				# draw from sech^2 distribution
				rng_max = self.sech2_cdf(self.absz_max, hz)
				rng_min = self.sech2_cdf(-self.absz_max, hz)
				rng = random.uniform(rng_min, rng_max)
				finalz = self.inverse_sech2_cdf(rng, hz)
				analog_id = -1
				self._file.write("%d\t%.2f\t%d\t%.2f\n" % (zone, tform,
					analog_id, finalz))
			else:
				pass
			return zone
		else:
			if self.post_process:
				if time < self.end_time:
					return zone
				else:
					R = Rform + self.dR
					return int((R / self.zone_width))
			else:
				# Interpolate between Rform and Rfinal at current time
				R = self.interpolator(Rform, Rform + self.dR, tform, time)
				# return int((R / self.zone_width) - 0.5) # ? reason for -0.5?
				return int((R / self.zone_width))

	def close_file(self):
		r"""
		Closes the output file - should be called after the multizone model
		simulation runs.
		"""
		self._file.close()

	@property
	def write(self):
		r"""
		Type : bool

		Whether or not to write out to the extra star particle data output
		file. For internal use by the vice.multizone object only.
		"""
		return self._write

	@write.setter
	def write(self, value):
		if isinstance(value, bool):
			self._write = value
		else:
			raise TypeError("Must be a boolean. Got: %s" % (type(value)))

	def interpolator(self, Rform, Rfinal, tform, time):
		r"""
		Interpolate between the formation and final radius following the fit
		to the h277 data, $\Delta R \propto t^{0.33}$.

		Parameters
		----------
		Rform : float
			Radius of formation in kpc.
		Rfinal : float
			Final radius in kpc.
		tform : float
			Formation time in Gyr.
		time : float
			Simulation time in Gyr.

		Returns
		-------
		float
			Radius at the current simulation time in kpc.
		"""
		tfrac = (time - tform) / (self.end_time - tform)
		return Rform + (Rfinal - Rform) * (tfrac**0.33)
		# return Rform + (Rfinal - Rform) * (tfrac**0.5)

	@staticmethod
	def migr_scale(age, Rform):
		r"""
		A prescription for $\sigma_{\Delta R}$, the scale of the Gaussian
		distribution of radial migration.

		$$ \sigma_{\Delta R} = 1.35 (R_\rm{form}/8\,\rm{kpc})^{0.61}
		(\tau/1\,\rm{Gyr})^{0.33} $$

		Parameters
		----------
		age : float or array-like
			Age of the stellar population in Gyr.
		Rform : float or array-like
			Formation radius of the stellar population in kpc.

		Returns
		-------
		float or array-like
			Scale factor for radial migration $\sigma_{\Delta R}$.
		"""
		return 1.35 * (age ** 0.33) * (Rform / 8) ** 0.61
		# return 1.82 * (age ** 0.33) * (Rform / 8) ** 0.61
		# return 1.82 * age**0.33

	@staticmethod
	def scale_height(age, Rfinal):
		r"""
		The scale height $h_z$ as a function of age and final radius:

		$$ h_z = (0.25\,{\rm kpc})
		\exp\Big(\frac{\tau-5\,{\rm Gyr}}{7.0\,{\rm Gyr}}\Big)
		\exp\Big(\frac{R_{\rm final}-8\,{\rm kpc}}{6.0\,{\rm kpc}}\Big) $$

		Parameters
		----------
		age : float or array-like
			Age in Gyr.
		Rfinal : float or array-like
			Final radius $R_{\rm{final}}$ in kpc.

		Returns
		-------
		float
			Scale height $h_z$ in kpc.
		"""
		return 0.25 * m.exp((age - 5.) / 7.) * m.exp((Rfinal - 8.) / 6.)
		# return 0.18 * (age ** 0.63) * (Rform / 8) ** 1.15

	@staticmethod
	def sech2_cdf(z, scale):
		r"""
		The cumulative distribution function (CDF) of the hyperbolic sec-square
		probability distribution function (PDF), which determines the density
		of stars as a function of distance from the midplane $z$. For some
		scaling $h_z$, the PDF is

		$$ \rm{PDF}(z) = \frac{1}{4 h_z} \cosh^{-2}\Big(\frac{z}{2 h_z}\Big) $$

		and the CDF is

		$$ \rm{CDF}(z) = \frac{1}{1 + e^{-z / h_z}} $$

		Parameters
		----------
		x : float
			Independent variable.
		scale : float
			Width of the sech-squared distribution, with the same units as x.

		Returns
		-------
		float
			The value of the CDF at the given x.
		"""
		return 1 / (1 + m.exp(-z / scale))

	@staticmethod
	def inverse_sech2_cdf(cdf, scale):
		r"""
		The inverse of the sech$^2$ CDF.

		For some scaling $h_z$, the $z$ corresponding to the given value of
		the CDF is

		$$ z = -h_z \ln\Big(\frac{1}{\rm{CDF}} - 1\Big) $$

		Parameters
		----------
		cdf : float
			The value of the CDF. Must be within the exclusive interval (0, 1).
		scale : float
			Width of the sech$^2$ distribution. Must be positive.

		Returns
		-------
		float
			The value of $z$ corresponding to the CDF value, in the same units
			as ``scale''.
		"""
		if cdf <= 0. or cdf >= 1.:
			raise ValueError("The value of the CDF must be between 0 and 1.")
		if scale <= 0.:
			raise ValueError("The scale height must be positive.")
		return -scale * m.log(1 / cdf - 1)


class churning:
	r"""

	Parameters
	----------
	radbins : array-like
		The bins in galactocentric radius in kpc corresponding to each annulus.
	zone_width : float [default : 0.1]
		Width of each radial zone in kpc.
	end_time : float [default : 13.2]
		The final simulation timestep in Gyr.
	filename : ``str`` [default : "stars.out"]
		The name of the file to write the extra star particle data to.
	absz_max : float [default : 3]
		Maximum |z|-height above the midplane in kpc. The default corresponds
		to the maximum value in the h277 sample.

	Attributes
	----------
	write : ``bool`` [default : False]
		A boolean describing whether or not to write to an output file when
		the object is called. The ``multizone`` object, and by extension the
		``milkyway`` object, automatically switch this attribute to True at the
		correct time to record extra data.

	Calling
	-------
	Returns the star's current zone at the given time.
	zone : int
		Birth zone of the star.
	tform : float
		Formation time of the star in Gyr.
	time : float
		Current simulation time in Gyr.
	"""
	def __init__(self, radbins, zone_width = ZONE_WIDTH, end_time = END_TIME,
			  filename = "stars_churn.out", absz_max = 3., post_process = False,
			  churning_probability=0.01, churning_R_drop=0.0005,
			  churning_scale=1., printing=False):
		self.radial_bins = radbins
		self.zone_width = zone_width
		self.end_time = end_time
		self.absz_max = absz_max
		self.post_process = post_process
		self.churning_probability = churning_probability
		self.churning_R_drop = churning_R_drop
		self.churning_scale = churning_scale
		self.printing = printing

		if isinstance(filename, str):
			self._file = open(filename, 'w')
			# Same format as above for compatibility
			self._file.write("# zone_origin\ttime_origin\tanalog_id\tzfinal\n")
		else:
			raise TypeError("Filename must be a string. Got: %s" % (
				type(filename)))

		# Multizone object automatically swaps this to True in setting up
		# its stellar population zone histories
		self.write = False

	def __call__(self, zone, tform, time):
		Rform = self.zone_width * (zone + 0.5)
# 		age = self.end_time - tform
		if tform == time:
			# particle just formed, don't apply migration
			self.R = Rform
			# Write to file
			if self.write:
				analog_id = -1
				self._file.write("%d\t%.2f\t%d\t\n" % (zone, tform, analog_id))
			else:
				pass
			return zone
		else:
			if self.post_process:
				if time < self.end_time:
					return zone
				else:
					return int((self.R / self.zone_width))
			else: # This is where churning is happening
				self.apply_churning()
				return int((self.R / self.zone_width))

	def close_file(self):
		r"""
		Closes the output file - should be called after the multizone model
		simulation runs.
		"""
		self._file.close()

	@property
	def write(self):
		r"""
		Type : bool

		Whether or not to write out to the extra star particle data output
		file. For internal use by the vice.multizone object only.
		"""
		return self._write

	@write.setter
	def write(self, value):
		if isinstance(value, bool):
			self._write = value
		else:
			raise TypeError("Must be a boolean. Got: %s" % (type(value)))

	def apply_churning(self):
		r"""
		calculate if churning happens and randomly draw a migration
		distance if it does
		"""
		churn_prob = (self.churning_probability -
				self.churning_R_drop * self.R) * 100000
		if random.randint(0, 100000) <= churn_prob:
			while True: # ensure 0 < R <= Rmax
				new_R = self.R + random.gauss(0, self.churning_scale)
				if new_R > 0 and new_R <= self.radial_bins[-1]*self.zone_width:
					self.R = new_R
					break

class blurring:
	r"""

	Parameters
	----------
	radbins : array-like
		The bins in galactocentric radius in kpc corresponding to each annulus.
	zone_width : float [default : 0.1]
		Width of each radial zone in kpc.
	end_time : float [default : 13.2]
		The final simulation timestep in Gyr.
	filename : ``str`` [default : "stars.out"]
		The name of the file to write the extra star particle data to.
	absz_max : float [default : 3]
		Maximum |z|-height above the midplane in kpc. The default corresponds
		to the maximum value in the h277 sample.

	Attributes
	----------
	write : ``bool`` [default : False]
		A boolean describing whether or not to write to an output file when
		the object is called. The ``multizone`` object, and by extension the
		``milkyway`` object, automatically switch this attribute to True at the
		correct time to record extra data.

	Calling
	-------
	Returns the star's current zone at the given time.
	zone : int
		Birth zone of the star.
	tform : float
		Formation time of the star in Gyr.
	time : float
		Current simulation time in Gyr.
	"""
	def __init__(self, radbins, zone_width = ZONE_WIDTH, end_time = END_TIME,
			  filename = "stars_blur.out", absz_max = 3., post_process = False,
			  blurring_prob=0.03, blurring_R_drop=0.0003,
			  blurring_active=False, start_ecc=0.0, ecc_increase=0.05,
			  max_blur_time=2, rotation_Gyr=0.250, printing=False):
		self.radial_bins = radbins
		self.zone_width = zone_width
		self.end_time = end_time
		self.absz_max = absz_max
		self.post_process = post_process
		self.blurring_prob = blurring_prob
		self.blurring_R_drop = blurring_R_drop
		self.blurring_active = blurring_active
		self.start_ecc = start_ecc
		self.max_blur_time= max_blur_time
		self.blurring_start = -1
		self.rotation_Gyr = rotation_Gyr
		self.ecc_increase = ecc_increase
		self.printing = printing
		self.timestep = -1

		if isinstance(filename, str):
			self._file = open(filename, 'w')
			# Same format as above for compatibility
			self._file.write("# zone_origin\ttime_origin\tanalog_id\tzfinal\n")
		else:
			raise TypeError("Filename must be a string. Got: %s" % (
				type(filename)))

		# Multizone object automatically swaps this to True in setting up
		# its stellar population zone histories
		self.write = False

	def __call__(self, zone, tform, time):
		self.R_G = self.zone_width * (zone + 0.5)
# 		age = self.end_time - tform
		if tform == time:
			self.ecc = self.start_ecc
			# particle just formed, don't apply migration
			if self.blurring_active == True:
				self.blurring_end_time = tform + (random.randint(0, 1000) /
								 1000) * self.max_blur_time
			# Write to file
			if self.write:
				analog_id = -1
				self._file.write("%d\t%.2f\t%d\t\n" % (zone, tform, analog_id))
			else:
				pass
			return zone
		else:
			if self.timestep == -1:
				self.timestep = time - tform

			if self.post_process:
				if time < self.end_time:
					return zone
				else:
					R_current = self.eccentric_orbit(time)
					return int((R_current / self.zone_width))
			else: # This is where blurring is happening
				self.apply_blurring(time)
				R_current = self.eccentric_orbit(time)
# 				print(self.R_G, R_current, self.ecc, time)
				return int((R_current / self.zone_width))

	def close_file(self):
		r"""
		Closes the output file - should be called after the multizone model
		simulation runs.
		"""
		self._file.close()

	@property
	def write(self):
		r"""
		Type : bool

		Whether or not to write out to the extra star particle data output
		file. For internal use by the vice.multizone object only.
		"""
		return self._write

	@write.setter
	def write(self, value):
		if isinstance(value, bool):
			self._write = value
		else:
			raise TypeError("Must be a boolean. Got: %s" % (type(value)))

	def apply_blurring(self, time):
		r"""
		calculate if blurring happens or, if active, increase the eccentricity
		"""
		if self.blurring_active is True:
			self.ecc = self.ecc + self.ecc_increase * self.timestep
			if time > self.blurring_end_time:
				self.blurring_active = False
		else:
			blur_prob = (self.blurring_prob -
					self.blurring_R_drop * self.R_G) * 100000
			if random.randint(0, 100000) <= blur_prob:
				self.blurring_active = True
				self.blurring_end_time = time + (random.randint(0, 1000) /
								 1000) * self.max_blur_time
				if self.blurring_start < 0:
					self.blurring_start = time

	def eccentric_orbit(self, time):
		r"""
		calculate where on the eccentric orbit the body currently is
		"""
		c = self.R_G * self.ecc
# 		print(c, time, self.ecc)

		new_R = self.R_G + c * m.sin(
			(time - self.blurring_start) / self.rotation_Gyr * 2 * m.pi)
		if new_R <= 0:
			return 0.1
		elif new_R > self.radial_bins[-1]*self.zone_width:
			return (self.radial_bins[-1] - 0.1) * self.zone_width
		else:
			return new_R



class migration_churn_blur:
	r"""

	Parameters
	----------
	radbins : array-like
		The bins in galactocentric radius in kpc corresponding to each annulus.
	zone_width : float [default : 0.1]
		Width of each radial zone in kpc.
	end_time : float [default : 13.2]
		The final simulation timestep in Gyr.
	filename : ``str`` [default : "stars.out"]
		The name of the file to write the extra star particle data to.
	absz_max : float [default : 3]
		Maximum |z|-height above the midplane in kpc. The default corresponds
		to the maximum value in the h277 sample.

	Attributes
	----------
	write : ``bool`` [default : False]
		A boolean describing whether or not to write to an output file when
		the object is called. The ``multizone`` object, and by extension the
		``milkyway`` object, automatically switch this attribute to True at the
		correct time to record extra data.

	Calling
	-------
	Returns the star's current zone at the given time.
	zone : int
		Birth zone of the star.
	tform : float
		Formation time of the star in Gyr.
	time : float
		Current simulation time in Gyr.
	"""
	def __init__(self, radbins, zone_width = ZONE_WIDTH, end_time = END_TIME,
			  filename = "stars.out", absz_max = 3., post_process = False,
			  churning_probability=0.01, churning_R_drop=0.0005,
			  churning_scale=1., blurring_prob=0.03, blurring_R_drop=0.0003,
			  blurring_active=False, start_ecc=0.0, ecc_increase=0.005,
			  max_blur_time=2, rotation_Gyr=0.250):
		self.radial_bins = radbins
		self.zone_width = zone_width
		self.end_time = end_time
		self.absz_max = absz_max
		self.post_process = post_process
		#initiallise churning subclass
		self.chur = churning(radbins, zone_width = zone_width,
					   end_time = end_time,
					   filename = "stars_chur.out", absz_max = absz_max,
					   post_process = post_process,
					   churning_probability=churning_probability,
					   churning_R_drop=churning_R_drop,
					   churning_scale=churning_scale)
		#initialise blurring subclass
		self.blur = blurring(radbins, zone_width = zone_width,
					   end_time = end_time,
					   filename = "stars_blur.out", absz_max = absz_max,
					   post_process = post_process,
					   blurring_prob=blurring_prob,
					   blurring_R_drop=blurring_R_drop,
					   blurring_active=blurring_active, start_ecc=start_ecc,
					   ecc_increase=ecc_increase, max_blur_time=max_blur_time,
					   rotation_Gyr=rotation_Gyr)

		if isinstance(filename, str):
			self._file = open(filename, 'w')
			# Same format as above for compatibility
			self._file.write("# zone_origin\ttime_origin\tanalog_id\tzfinal\n")
		else:
			raise TypeError("Filename must be a string. Got: %s" % (
				type(filename)))

		# Multizone object automatically swaps this to True in setting up
		# its stellar population zone histories
		self.write = False

	def __call__(self, zone, tform, time):
		Rform = self.zone_width * (zone + 0.5)
# 		age = self.end_time - tform
		if tform == time:
			# particle just formed, don't apply migration
			self.R_G = Rform

			#need to execute this function for the first time,
 			#values shouldn't change
			self.R_G = self.chur(zone, tform, time) * self.zone_width
			R = self.blur(int((self.R_G / self.zone_width)),
			  tform, time) * self.zone_width
			# Write to file
			if self.write:
				analog_id = -1
				self._file.write("%d\t%.2f\t%d\t\n" % (zone, tform, analog_id))
			else:
				pass
			return zone
		else:
			if self.post_process:
				if time < self.end_time:
					return zone
				else:
					return int((self.R / self.zone_width))
			else: # This is where churning and blurring are happening
				self.R_G = self.chur(zone, tform, time) * self.zone_width
				R = self.blur(int((self.R_G / self.zone_width)),
				  tform, time) * self.zone_width
				print(self.R_G, R)
				return int((R / self.zone_width))

