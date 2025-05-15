def particle_string(PID, status, mother1,mother2,color1,color2,px,py,pz,E, mass,vtim,helicity):
    """Returns a LHE event entry corresponding to a single particle"""
    return " %8d %2d %4d %4d %4d %4d %+13.10e %+13.10e %+13.10e %14.10e %14.10e %10.4e %10.4e" \
            % (PID,
               status,
               mother1,
               mother2,
               color1,
               color2,
               px,
               py,
               pz,
               E,
               mass,
               vtim,
               helicity)

def create_xml_event(event, ma, mN):
    """
    Constructs an LHE-style event entry string
    Args:
        event: A (5,4) array of 4-momenta with each corresponding to incoming
        photon, ALP, Nucleus, daughter1, daughter2 and outgoing electron.
    Returns:
        event_string = output string including <event> tags
    """
    # compulsory event information needed for LHE parsers
    # see p. 6 of https://arxiv.org/pdf/hep-ph/0109068.pdf
    #print("final event ",event[5,1],event[5,2],event[5,3],event[5,0])
    evt_info = " {num_part} {proc_id} {weight} {scale} {qed_coupling} {qcd_coupling}\n".format(num_part=6,proc_id=2,weight=1,scale=-1,qed_coupling=1./137.,qcd_coupling=0.1081)
    photon = particle_string(PID=22, status = -1, mother1 =0,mother2 =0,color1=0,color2=0,px=event[0,1],py=event[0,2],pz=event[0,3],E=event[0,0], mass=0,vtim=0,helicity=0) + '\n'
    nucleus = particle_string(PID=623, status = -1, mother1 =0,mother2 =0,color1=0,color2=0,px=0,py=0,pz=0,E=mN, mass=mN,vtim=0,helicity=0)+ '\n'
    dv = event[5] # ALP decay vertex
    # use vtim field to store the decay time in cm of the ALP. Use this to reconstruct the displaced vertex
    # example: 11  1    1    2    0    0 px py pz e me 0.0000e+00 1.0000e+00
    electron=particle_string(PID=11, status = 1, mother1 =1,mother2 =2,color1=0,color2=0,px=event[6,1],py=event[6,2],pz=event[6,3],E=event[6,0], mass=5.1099890000e-04,vtim=0,helicity=1) + '\n'
    ALP = particle_string(PID=666, status = 2, mother1 =1,mother2 =2,color1=0,color2=0,px=event[1,1],py=event[1,2],pz=event[1,3],E=event[1,0], mass=ma,vtim=dv[0],helicity=0)+ '\n'
    nucleus_prime = particle_string(PID=623, status = 1, mother1 =0,mother2 =0,color1=0,color2=0,px=event[2,1],py=event[2,2],pz=event[2,3],E=event[2,0], mass=mN,vtim=0,helicity=0)+ '\n'

    daughter1 = particle_string(PID=22, status = 1, mother1 =3,mother2 =0,color1=0,color2=0,px=event[3,1],py=event[3,2],pz=event[3,3],E=event[3,0], mass=0,vtim=0,helicity=0) + '\n'
    daughter2 = particle_string(PID=22, status = 1, mother1 =3,mother2 =0,color1=0,color2=0,px=event[4,1],py=event[4,2],pz=event[4,3],E=event[4,0], mass=0,vtim=0,helicity=0) + '\n'
    decay_vertex = '#vertex ' + str(dv[1]) + ' ' + str(dv[2]) + ' ' + str(dv[3]) + ' [' + str(dv[0]) + '] ' + '\n'

    event_string = "<event>\n" + evt_info + photon + nucleus  + ALP + nucleus_prime + electron + daughter1 + daughter2 + decay_vertex + "</event>"

    return event_string

def create_xml_event_target(event, ma, mN):
    """
    Constructs an LHE-style event entry string
    Args:
        event: A (5,4) array of 4-momenta with each corresponding to incoming
        photon, ALP, Nucleus, daughter1, daughter2 and outgoing electron.
    Returns:
        event_string = output string including <event> tags
    """
    # compulsory event information needed for LHE parsers
    # see p. 6 of https://arxiv.org/pdf/hep-ph/0109068.pdf
    print("target")
    evt_info = " {num_part} {proc_id} {weight} {scale} {qed_coupling} {qcd_coupling}\n".format(num_part=6,proc_id=2,weight=1,scale=-1,qed_coupling=1./137.,qcd_coupling=0.1081)
    photon = particle_string(PID=22, status = -1, mother1 =0,mother2 =0,color1=0,color2=0,px=event[0,1],py=event[0,2],pz=event[0,3],E=event[0,0], mass=0,vtim=0,helicity=0) + '\n'
    nucleus = particle_string(PID=623, status = -1, mother1 =0,mother2 =0,color1=0,color2=0,px=0,py=0,pz=0,E=mN, mass=mN,vtim=0,helicity=0)+ '\n'
    electron=particle_string(PID=11, status = 1, mother1 =1,mother2 =2,color1=0,color2=0,px=event[6,1],py=event[6,2],pz=event[6,3],E=event[6,0], mass=5.1099890000e-04,vtim=0,helicity=1) + '\n'
    dv = event[5]
    ALP = particle_string(PID=666, status = 2, mother1 =1,mother2 =2,color1=0,color2=0,px=event[1,1],py=event[1,2],pz=event[1,3],E=event[1,0], mass=ma,vtim=dv[0],helicity=0)+ '\n'
    nucleus_prime = particle_string(PID=623, status = 1, mother1 =0,mother2 =0,color1=0,color2=0,px=event[2,1],py=event[2,2],pz=event[2,3],E=event[2,0], mass=mN,vtim=0,helicity=0)+ '\n'
    decay_vertex = '#vertex ' + str(0)+ ' ' + str(0) + ' ' + str(0) + ' [' + str(0) + '] ' + '\n'
    event_string = "<event>\n" + evt_info + photon + nucleus  + ALP + nucleus_prime + electron + decay_vertex +"</event>"
    return event_string

def create_xml_event_decay(event, ma, mN):
    """
    Constructs an LHE-style event entry string
    Args:
        event: A (5,4) array of 4-momenta with each corresponding to incoming
        photon, ALP, Nucleus, daughter1, daughter2 and outgoing electron.
    Returns:
        event_string = output string including <event> tags
    """
    print("decay")
    # compulsory event information needed for LHE parsers
    # see p. 6 of https://arxiv.org/pdf/hep-ph/0109068.pdf
    #print("final event ",event[5,1],event[5,2],event[5,3],event[5,0])
    evt_info = " {num_part} {proc_id} {weight} {scale} {qed_coupling} {qcd_coupling}\n".format(num_part=6,proc_id=2,weight=1,scale=-1,qed_coupling=1./137.,qcd_coupling=0.1081)
    dv = event[5] # ALP decay vertex
    # use vtim field to store the decay time in cm of the ALP. Use this to reconstruct the displaced vertex
    daughter1 = particle_string(PID=22, status = 1, mother1 =3,mother2 =0,color1=0,color2=0,px=event[3,1],py=event[3,2],pz=event[3,3],E=event[3,0], mass=0,vtim=0,helicity=0) + '\n'
    daughter2 = particle_string(PID=22, status = 1, mother1 =3,mother2 =0,color1=0,color2=0,px=event[4,1],py=event[4,2],pz=event[4,3],E=event[4,0], mass=0,vtim=0,helicity=0) + '\n'
    decay_vertex = '#vertex ' + str(dv[1]) + ' ' + str(dv[2]) + ' ' + str(dv[3]) + ' [' + str(dv[0]) + '] ' + '\n'
    event_string = "<event>\n" + evt_info + daughter1 + daughter2 + decay_vertex + "</event>"

    return event_string


def create_LHE_file(ma, mN, events, filename_target, filename_decay, header_str=""):
    """
    Writes a list of events to a LHE file
    Args:
        ma: Mass of the ALP
        mN: Mass of the target Nucleus
        events:  A (N, 5, 4) array of 4-momenta with each corresponding to incoming
            photon, ALP, Nucleus, daughter1, daughter2 for N events.
        filename: String of the output filename

    Returns:
        Nothing. output is written into filename.lhe
    """
    LHE_file_target = header_str
    LHE_file_target += create_xml_event_target(events[0], ma, mN)

    for i in range(len(events)-1):
        LHE_file_target += '\n'
        LHE_file_target += create_xml_event_target(events[i+1], ma, mN)

    text_file_target = open(filename_target, "w")
    n_target = text_file_target.write(LHE_file_target)
    text_file_target.close()

    LHE_file_decay = header_str
    LHE_file_decay += create_xml_event_decay(events[0], ma, mN)

    for i in range(len(events)-1):
        LHE_file_decay += '\n'
        LHE_file_decay += create_xml_event_decay(events[i+1], ma, mN)

    text_file_decay = open(filename_decay, "w")
    n_decay = text_file_decay.write(LHE_file_decay)
    text_file_decay.close()

    return
