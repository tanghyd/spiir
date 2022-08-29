#
# Copyright (C) 2018 Qi Chu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.



from __future__ import absolute_import
from __future__ import print_function
import os
import re
import numpy
import scipy
from glue.ligolw import ligolw, lsctables, array, param, utils, types

import lal
import lalsimulation
from gstlal_spiir.spiirbank import cbc_template_iir
from gstlal import cbc_template_fir
from six.moves import range

# FIXME:  require calling code to provide the content handler
class DefaultContentHandler(ligolw.LIGOLWContentHandler):
    pass
array.use_in(DefaultContentHandler)
param.use_in(DefaultContentHandler)
lsctables.use_in(DefaultContentHandler)

def get_bankid_from_bankname(bankname):
    tmp_name = os.path.split(bankname)[-1]
    tmp_name = re.sub(r'[HLV]1', '', tmp_name)
    search_result = re.search(r'\d{1,4}', tmp_name)
    try:
        bankid = search_result.group()
    except:
        raise ValueError("bankid should be the first 3/4 digits of the given name, could not find the digits from %s" % tmp_name)

    bankid_strip = bankid.lstrip('0')
    if bankid_strip is '':
        return 0
    else:
        return int(bankid_strip)

def parse_iirbank_string(bank_string):
    """
    parses strings of form 

    H1:bank1.xml,H2:bank2.xml,L1:bank3.xml,H2:bank4.xml,... 

    into a dictionary of lists of bank files.
    """
    out = {}
    if bank_string is None:
        return out
    for b in bank_string.split(','):
        ifo, bank = b.split(':')
        out.setdefault(ifo, []).append(bank)
    return out



def get_maxrate_from_xml(filename, contenthandler = DefaultContentHandler, verbose = False):
    xmldoc = utils.load_filename(filename, contenthandler = contenthandler, verbose = verbose)

    for root in (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == "gstlal_iir_bank_Bank"):

        sample_rates = [int(float(r)) for r in param.get_pyvalue(root, 'sample_rate').split(',')]

    return max(sample_rates)


def get_negative_from_xml(filename, contenthandler = DefaultContentHandler, verbose = False):
    xmldoc = utils.load_filename(filename, contenthandler = contenthandler, verbose = verbose)
    for root in (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == "gstlal_iir_bank_Bank"):
        negative_latency = int(param.get_pyvalue(root, 'negative_latency'))

    return negative_latency

# a modification from the cbc_template_fir.generate_templates
def gen_whitened_fir_template(
        template_table, approximant, irow, psd, f_low, time_slices, 
        autocorrelation_length = 201, sampleRate = 4096., 
        negative_latency = 0, verbose = False):

    """!
    Generate a bank of templates, which are
    whitened with a given psd.
    """
    sample_rate_max = sampleRate
    duration = max(time_slices['end'])
    length_max = int(round(duration * sample_rate_max))

    # Some input checking to avoid incomprehensible error messages
    if not template_table:
        raise ValueError("template list is empty")
    if f_low < 0.:
        raise ValueError("f_low must be >= 0.: %s" % repr(f_low))

    # working f_low to actually use for generating the waveform.  pick
    # template with lowest chirp mass, compute its duration starting
    # from f_low;  the extra time is 10% of this plus 3 cycles (3 /
    # f_low);  invert to obtain f_low corresponding to desired padding.
    # NOTE:  because SimInspiralChirpStartFrequencyBound() does not
    # account for spin, we set the spins to 0 in the call to
    # SimInspiralChirpTimeBound() regardless of the component's spins.
    template = min(template_table, key = lambda row: row.mchirp)
    tchirp = lalsimulation.SimInspiralChirpTimeBound(
        f_low, template.mass1 * lal.MSUN_SI, template.mass2 * lal.MSUN_SI, 0., 0.)
    working_f_low = lalsimulation.SimInspiralChirpStartFrequencyBound(
        1.1 * tchirp + 3. / f_low, template.mass1 * lal.MSUN_SI, template.mass2 * lal.MSUN_SI)

    # Add duration of PSD to template length for PSD ringing, 
    # round up to power of 2 count of samples
    working_length = cbc_template_iir.ceil_pow_2(length_max + round(1./psd.deltaF * sample_rate_max))
    working_duration = float(working_length) / sample_rate_max

    # Smooth the PSD and interpolate to required resolution
    if psd is not None:
        psd = cbc_template_fir.condition_psd(
            psd, 1.0 / working_duration, minfs = (working_f_low, f_low), 
            maxfs = (sample_rate_max / 2.0 * 0.90, sample_rate_max / 2.0))

    if verbose:
        logging.basicConfig(format='%(asctime)s %(message)s', level = logging.DEBUG)
        logging.info("working_f_low %f, working_duration %f, flower %f,", \
            "sampleRate %f" % (working_f_low, working_duration, f_low, sampleRate))
    revplan = lal.CreateReverseCOMPLEX16FFTPlan(working_length, 1)
    fwdplan = lal.CreateForwardREAL8FFTPlan(working_length, 1)
    tseries = lal.CreateCOMPLEX16TimeSeries(
        name = "timeseries",
        epoch = lal.LIGOTimeGPS(0.),
        f0 = 0.,
        deltaT = 1.0 / sample_rate_max,
        length = working_length,
        sampleUnits = lal.Unit("strain")
    )
    fworkspace = lal.CreateCOMPLEX16FrequencySeries(
        name = "template",
        epoch = lal.LIGOTimeGPS(0),
        f0 = 0.0,
        deltaF = 1.0 / working_duration,
        length = working_length // 2 + 1,
        sampleUnits = lal.Unit("strain s")
    )
    # Multiply by 2 * length of the number of sngl_inspiral 
    # rows to get the sine/cosine phases.
    template_bank = [numpy.zeros((2 * len(template_table), int(round(rate*(end-begin)))), dtype = "double") for rate,begin,end in time_slices]

    # Store the original normalization of the waveform.  After
    # whitening, the waveforms are normalized.  Use the sigmasq factors
    # to get back the original waveform.
    sigmasq = []

    # Generate each template, downsampling as we go to save memory
    max_ringtime = max([chirptime.ringtime(row.mass1*lal.MSUN_SI + row.mass2*lal.MSUN_SI, chirptime.overestimate_j_from_chi(max(row.spin1z, row.spin2z))) for row in template_table])
    row = template_table[irow]
    if verbose:
        print("generating template %d/%d:  m1 = %g, m2 = %g, s1x = %g,", \
           "s1y = %g, s1z = %g, s2x = %g, s2y = %g, s2z = %g, sample rate", \
            "%d, working_duration %f" % (irow + 1, len(template_table), 
            row.mass1, row.mass2, row.spin1x, row.spin1y, row.spin1z, 
            row.spin2x, row.spin2y, row.spin2z, sample_rate_max, working_duration))

    #
    # generate "cosine" component of frequency-domain template.
    # waveform is generated for a canonical distance of 1 Mpc.
    #

    fseries = cbc_template_fir.generate_template(
        row, approximant, sample_rate_max, working_duration, f_low, 
        sample_rate_max / 2., fwdplan = fwdplan, fworkspace = fworkspace)

    #
    # whiten and add quadrature phase ("sine" component)
    #

    if psd is not None:
        lal.WhitenCOMPLEX16FrequencySeries(fseries, psd)
    fseries = add_quadrature_phase(fseries, working_length)

    #
    # compute time-domain autocorrelation function
    #
    # Check parity of autocorrelation length
    if autocorrelation_length is not None:
        if not (autocorrelation_length % 2):
            raise ValueError("autocorrelation_length must be odd (got %d)" % autocorrelation_length)
    autocorrelation_bank_full = numpy.zeros(autocorrelation_length, dtype = "cdouble")


    autocorrelation = normalized_autocorrelation(fseries, revplan).data.data
    autocorrelation_bank_full[::-1] = numpy.concatenate(
        (autocorrelation[-(autocorrelation_length // 2):], 
        autocorrelation[:(autocorrelation_length // 2  + 1)]))

    #
    # transform template to time domain
    #

    lal.COMPLEX16FreqTimeFFT(tseries, fseries, revplan)

    data_full = tseries.data.data
    epoch_time = fseries.epoch.gpsSeconds + fseries.epoch.gpsNanoSeconds*1.e-9
    #
    # extract the portion to be used for filtering
    #


    #
    # condition the template if necessary (e.g. line up IMR
    # waveforms by peak amplitude)
    #

    # Use our own condition_IMR_templates to ajust the end time to be the merger time
    if approximant in gstlal_IMR_approximants:
        data_full, target_index = condition_imr_template(
            approximant, data_full, epoch_time, sample_rate_max, max_ringtime)
        # record the new end times for the waveforms (since we performed the shifts)
        row.end = lal.LIGOTimeGPS(float(target_index-(len(data_full) - 1.))/sample_rate_max)
    else:
        data_full *= cbc_template_iir.tukeywindow(data_full, samps = 32)

    data = data_full[-length_max:-int(1+negative_latency*sampleRate)]

    # This is to normalize whitened template so it = h_{whitened at 1MPC}(t)
    # NOTE: because
    # XLALWhitenCOMPLEX16FrequencySeries() computed
    #
    # \tilde{h}'_{k} = \sqrt{2 \Delta f} \tilde{h}_{k} / \sqrt{S_{k}}
    # need to devide the time domain whitened waveform by \sqrt{2 \Delta f}
    data /= numpy.sqrt(2./working_duration)

    #
    # normalize so that inner product of template with itself
    # is 2
    #

    #norm = abs(numpy.dot(data, numpy.conj(data)))
    #data *= cmath.sqrt(2 / norm)
    print("template length %d" % len(data))

    autocorrelation_bank = normalized_crosscorr(data_full, data, autocorrelation_length)
    return data, data_full, autocorrelation_bank, autocorrelation_bank_full


def gen_whitened_spiir_template_and_reconstructed_waveform(sngl_inspiral_table, approximant, irow, psd, sampleRate = 4096, waveform_domain = "FD", epsilon = 0.02, epsilon_min = 0.0, alpha = .99, beta = 0.25, flower = 30, autocorrelation_length = 201, req_min_match = 0.99, negative_latency = 0, verbose = False):

    working_state = gen_template_working_state(sngl_inspiral_table, flower, sampleRate = sampleRate)
    # Smooth the PSD and interpolate to required resolution
    if psd is not None:
        psd = cbc_template_fir.condition_psd(
                            psd,
                            1.0 / working_state["working_duration"],
                            minfs = (working_state["working_f_low"], flower),
                            maxfs = (sampleRate / 2.0 * 0.90, sampleRate / 2.0)
                            )

    # This is to avoid nan amp when whitening the amp
    #tmppsd = psd.data
    #tmppsd[numpy.isinf(tmppsd)] = 1.0
    #psd.data = tmppsd

    if verbose:
        logging.basicConfig(format='%(asctime)s %(message)s', level = logging.DEBUG)
        logging.info("condition of psd finished")
        logging.info("working_f_low %f, working_duration %f, flower %f, sampleRate %f" % (working_state["working_f_low"], working_state["working_duration"], flower, sampleRate))

    #
    # FIXME: condition the template if necessary (e.g. line up IMR
    # waveforms by peak amplitude)
    #

    original_epsilon = epsilon
    epsilon_increment = 0.001
    row = sngl_inspiral_table[irow]
    this_tchirp = lalsimulation.SimInspiralChirpTimeBound(
        flower, row.mass1 * lal.MSUN_SI, row.mass2 * lal.MSUN_SI, row.spin1z, row.spin2z)

    if verbose:
        logging.info("working_duration %f, chirp time %f" % (working_state["working_duration"], this_tchirp))

    # data = the cutted template. 
    # cut at the beginning to avoid long low SNR accumulation
    # cut at the end for negative latency template
    # data_full = original uncut template
    # fhigh is the estimated end frequency of data
    amp, phase, data, data_full, epoch_index, fhigh = gen_whitened_amp_phase(
        psd,
        approximant,
        waveform_domain,
        sampleRate,
        flower,
        working_state,
        row,
        is_frequency_whiten = 1,
        snr_cut = 0.998,
        negative_latency = negative_latency,
        verbose = verbose)

    # get the padded length, so SPIIR approximated waveform u_rev_pad
    # the original cut template h_pad, and the original one will be
    # padded to the same length
    pad_length = ceil_pow_2(len(data_full) + autocorrelation_length)


    # This is to normalize whitened template so it = h_{whitened at 1MPC}(t)
    # NOTE: because
    # XLALWhitenCOMPLEX16FrequencySeries() computed
    #
    # \tilde{h}'_{k} = \sqrt{2 \Delta f} \tilde{h}_{k} / \sqrt{S_{k}}
    # need to devide the time domain whitened waveform by \sqrt{2 \Delta f}
    amp /= numpy.sqrt(2./working_state["working_duration"])

    spiir_match = -1
    n_filters = 0
    nround = 1

    while(spiir_match < req_min_match and epsilon > epsilon_min and n_filters < 2000):
        a1, b0, delay, u_rev_pad = cbc_template_iir.gen_spiir_coeffs(amp, phase, pad_length, epsilon = epsilon)
        # get the cut waveform
        h_pad = numpy.zeros(pad_length * 1, dtype=numpy.cdouble)
        h_pad[-len(data):] = data

        # compute the SNR
        # deprecated: spiir_match = abs(numpy.dot(u_rev_pad, numpy.conj(h_pad_real)))
        # the following definition is more close to the reality
        norm_u = abs(numpy.dot(u_rev_pad, numpy.conj(u_rev_pad)))
        norm_h = abs(numpy.dot(h_pad, numpy.conj(h_pad)))
        norm_data_full = abs(numpy.dot(data_full, numpy.conj(data_full)))

        # overlap of spiir reconstructed waveform with template (spiir_template)
        spiir_match = abs(numpy.dot(u_rev_pad, numpy.conj(h_pad))/numpy.sqrt(norm_u * norm_h))
        # FIXME:normalize so that the SNR would match the expected SNR, 
        # using norm_h instead of norm_data_full ?
        b0 *= numpy.sqrt(norm_data_full / norm_u) * spiir_match
        n_filters = len(delay)

        if verbose:
            logging.info("number of rounds %d, epsilon %f,", \
                "spiir overlap with template %f, number of filters %d" % \
                (nround, epsilon, spiir_match, n_filters))


        if(nround == 1):
            original_match = spiir_match
            original_filters = len(a1)

        if(spiir_match < req_min_match):
            epsilon -= epsilon_increment

        nround += 1
    if verbose:
        logging.info("norm of the  template h_pad %f, norm of spiir response u_rev_pad %f" % (norm_h, norm_u))

    # normalize u_rev_pad so its square root of inner product is sqrt(norm_data_full) * spiir_match
    u_rev_pad = u_rev_pad * numpy.sqrt(norm_h / norm_u) * spiir_match

    return u_rev_pad, h_pad, data_full, fhigh

def matched_filt(template, strain, sampleRate = 4096.0):

    '''
    matched filtering using numpy fft
    template: complex
    data: time, real value
    The template is produced from using the gen_whitened_fir_template or 
    gen_whitened_spiir_template_and_reconstructed_waveform.
    The unit of template is s^-1/2. the data is generated from gstlal_whiten 
    where the unit is dimensionless.
    It needs to be normalized so the unit is s^-1/2, 
    same as the template for unit consistency.
    '''
    # check if the strain is continous
    last_time = strain[0, 0]
    dt = strain[1, 0] - last_time
    for i in range(1, len(strain[:,0])):
        this_time = strain[i, 0]
        if this_time - last_time > 2*dt:
            raise ValueError("Strain data is not continous, needs padding.", \
                "See function pad_gap in gstlal_matched_filter")
        last_time = this_time

    # data is generated from gstlal_play --whiten
    # where the unit is dimensionless.
    # It needs to be normalized so the unit is s^-1/2, i.e., *1/sqrt(dt).

    time = strain[:, 0]
    data = strain[:, 1]
    data /= numpy.sqrt(2.0/sampleRate)
    # if the template inner product is normalized to 2, 
    # need to convert its unit by doing the following:
    # template /= numpy.sqrt(2.0/sampleRate)
    # need to extend to 2 times to avoid cyclic artifacts
    working_length = max(len(template), len(data)) * 2
    template_len = len(template)
    fs = float(sampleRate)
    df = 1.0/ (working_length/ fs)
    template_pad = numpy.zeros(working_length, dtype = "cdouble")
    template_pad[:len(template)] = template
    data_pad = numpy.zeros(working_length, dtype = "double")
    data_pad[:len(data)] = data

    data_pad *= cbc_template_iir.tukeywindow(data_pad, samps = 32.)

    data_fft = numpy.fft.fft(data_pad)/ fs # times dt
    template_fft = numpy.fft.fft(template_pad)/ fs

    snr_fft = data_fft * template_fft.conjugate()
    # times df then the unit is dimensionless, 
    # default ifft has the output scaled by 1/N
    snr_time_conj = 2 * numpy.fft.ifft(snr_fft) * fs 
    # note that the snr needs to be conjugate as the template is conjugate
    snr_time = snr_time_conj.conjugate()
    sigmasq = (template_fft * template_fft.conjugate()).sum() * df
    sigma = numpy.sqrt(abs(sigmasq))
    # normalize snr
    snr_time /= sigma
    # need to shift the SNR because cross-correlation FFT plays integration 
    # on cyclic template_pad,
    # that the first value of snr_time is out(0) = data(0:) times template(0:). 
    # we actually need out(0) = data(0:) times template(-1)
    # for the first value where N is the len of template, 
    # so that out(N-1) = data(0:) times template (-N:)
    roll_len = template_len - 1 # note here is N - 1
    snr_time = numpy.roll(snr_time, roll_len)
    # find the time and SNR value at maximum:
    SNR = abs(snr_time)
    indmax = numpy.argmax(SNR)
    try:
        timemax = time[indmax]
    except:
        raise ValueError("max SNR is outside the data, need to collect more data for a more correct SNR estimation")
    f_ticks = numpy.linspace(1, working_length + 1, working_length) * df
    return snr_time, sigma, indmax, timemax, f_ticks, data_fft, template_fft


