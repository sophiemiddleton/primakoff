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
        photon, ALP, Nucleus, daughter1, daughter2.
    Returns:
        event_string = output string including <event> tags
    """
    # compulsory event information needed for LHE parsers
    # see p. 6 of https://arxiv.org/pdf/hep-ph/0109068.pdf
    evt_info = " {num_part} {proc_id} {weight} {scale} {qed_coupling} {qcd_coupling}\n".format(num_part=6,proc_id=2,weight=1,scale=-1,qed_coupling=1./137.,qcd_coupling=0.1081)
    photon = particle_string(PID=22, status = -1, mother1 =0,mother2 =0,color1=0,color2=0,px=event[0,1],py=event[0,2],pz=event[0,3],E=event[0,0], mass=0,vtim=0,helicity=0) + '\n'
    nucleus = particle_string(PID=623, status = -1, mother1 =0,mother2 =0,color1=0,color2=0,px=0,py=0,pz=0,E=mN, mass=mN,vtim=0,helicity=0)+ '\n'

    ALP = particle_string(PID=666, status = 2, mother1 =1,mother2 =2,color1=0,color2=0,px=event[1,1],py=event[1,2],pz=event[1,3],E=event[1,0], mass=ma,vtim=0,helicity=0)+ '\n'
    nucleus_prime = particle_string(PID=623, status = 1, mother1 =0,mother2 =0,color1=0,color2=0,px=event[2,1],py=event[2,2],pz=event[2,3],E=event[2,0], mass=mN,vtim=0,helicity=0)+ '\n'

    daughter1 = particle_string(PID=22, status = 1, mother1 =3,mother2 =0,color1=0,color2=0,px=event[3,1],py=event[3,2],pz=event[3,3],E=event[3,0], mass=0,vtim=0,helicity=0) + '\n'
    daughter2 = particle_string(PID=22, status = 1, mother1 =3,mother2 =0,color1=0,color2=0,px=event[4,1],py=event[4,2],pz=event[4,3],E=event[4,0], mass=0,vtim=0,helicity=0) + '\n'

    event_string = "<event>\n" + evt_info + photon + nucleus + ALP + nucleus_prime + daughter1 + daughter2 + "</event>"

    return event_string

def create_LHE_file(ma, mN, events, filename, header_str=""):
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
    LHE_file = header_str 
    LHE_file += create_xml_event(events[0], ma, mN)

    for i in range(len(events)-1):
        LHE_file += '\n'
        LHE_file += create_xml_event(events[i+1], ma, mN)

    text_file = open(filename, "w")
    n = text_file.write(LHE_file)
    text_file.close()

    return
