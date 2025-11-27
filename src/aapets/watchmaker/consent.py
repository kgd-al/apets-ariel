from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QHBoxLayout, QCheckBox, QSizePolicy, QLineEdit


class ConsentDialog(QDialog):
    def __init__(self, output_folder):
        super().__init__()
        self.setWindowTitle("Consent Dialog")

        self.output_folder = output_folder

        v_layout = QVBoxLayout()

        self.consent_text = QLabel("""
           <h1>Informed Consent Agreement</h1>
            <p>Before taking part in this study, please read the following information and indicate below whether you 
            consent to the conditions of participation.</p>
            
            <p>This study involves participation in interactive experiments using Ariel. The experiment typically takes
             15-30 minutes to complete, depending on the specific experiment configuration.</p>
            
            <p>During the experiment, we will collect data about your interactions with the AI system, including your
             actions, decisions, and responses. This data will be used solely for research purposes to improve our
              understanding of human-AI collaboration in reinforcement learning settings.</p>
            
            <p>Only the research team will have access to the data, and all the information is kept in secure locations
             approved by The Hybrid Intelligence Centre.</p>
            
            <div class="row justify-content-evenly">
                <div class="col-auto">
                    <p>
                        <strong>Principal Investigator:</strong>
                        <br>
                        Hybrid Intelligence Centre
                        <br>
                        The Netherlands
                    </p>
                </div>
            </div>
            
            <p>Participation is <b>completely voluntary</b>. You can decide to withdraw at any time and for any reason,
             including after participating. However, if you choose to withdraw after participating, we will not be
              required to undo the processing of your data that has taken place up until that time. The personal data we 
              have obtained from you up until the time when you withdraw your consent will be erased (where personal data
               is any data that can be linked to you, so this excludes any already anonymized data).</p>
            
            <p>We intend to report <b>the results of this study</b> in publications and/or presentations. Upon request,
             we can share these results with you through email. We also intend to share collected survey data with the
             broader research community, following FAIR (Findable, Accessible, Interoperable, Reusable) data
             principles. Any information from the study that could identify you as an individual will be replaced or
             removed before publishing the survey data. Personal characteristics are only collected in broad categories.
              Therefore, <b>your identity as a participant will remain confidential at all times</b>, and the risk of
               participation will be minimal. The data will be stored for 10 years.</p>
            
            <p>This project adheres to the <b>ethical guidelines</b> established by The Hybrid Intelligence Centre, and
             falls under fundamental research without any commercial purpose nor external stakeholders or partners. For
             details of our legal basis for using personal data and the rights you have over your data, please see the
             privacy information at 
             <a href="https://www.hybrid-intelligence-centre.nl/wp-content/uploads/2023/04/HI-Ethics-Policy.pdf" target="_blank">HI Centre Ethics Policy (PDF)</a>.</p>
            
            <p>For more information regarding this project, please contact:</p>
            
            <div class="row">
                <div class="col">
                    <p>
                        Hybrid Intelligence Center<br>
                        The Netherlands<br>
                        <a href="https://hybrid-intelligence-centre.nl" target="_blank">https://hybrid-intelligence-centre.nl</a>
                    </p>
                </div>
            </div>
        </div>
    </div> 
        """)
        self.consent_text.setWordWrap(True)
        v_layout.addWidget(self.consent_text)

        h_layout = QHBoxLayout()
        v_sublayout = QVBoxLayout()

        self.username = QLineEdit()
        self.username.setPlaceholderText("(Anonymous) Username")
        v_sublayout.addWidget(self.username)

        self.username_error = QLabel()
        self.username_error.setStyleSheet("color: red")
        v_sublayout.addWidget(self.username_error)

        h_layout.addLayout(v_sublayout)

        self.consent_box = QCheckBox("I consent to participating in the study")
        h_layout.addWidget(self.consent_box)

        v_layout.addLayout(h_layout)
        self.setLayout(v_layout)

        self.setFixedSize(v_layout.sizeHint())

        self.consent_box.toggled.connect(self.accept)
        self.username.textChanged.connect(self.check_username)

    def check_username(self):
        username = self.username.text()
        error = ""
        if len(username) < 3:
            error = "Username must be at least 3 characters long<red>"
        elif self.output_folder.joinpath(username).exists():
            error = "Username already taken"

        self.username_error.setText(error)

        ok = (len(error) == 0)
        self.consent_box.setEnabled(ok)
