import React from 'react';

export default class Navbar extends React.Component {
  componentDidMount() {
    $('.ui.dropdown').dropdown({
      direction: 'upward'
    });
  }
  render() {
    return (
      <div className="ui vertical inverted menu fixed" id="sidebar">
        <div className="item">
          <b className="team-menu">scotch</b>
        </div>
        <div className="item listings_channels">
          <div className="header listings_header">Channels</div>
          <form className="ui form" onSubmit={this.handleCreateChannel}>
            <div className="field">
              <input type="text" ref="newChannel" placeholder="Create Channel"></input>
            </div>
          </form>
          <div className="menu channel_list">
            Gigster Automated Sales Engineer
          </div>
        </div>
        <div className="item listings_direct-messages">
          <div className="header">Direct Messages</div>
        </div>
        <div className="ui inverted dropdown item user-menu">
          <div className="menu" id="userMenu">
            <div className="item">Sign In</div>
            <div className="item">Register</div>
          </div>
          <img className="ui avatar image user-menu_profile-pic" src="/Avatar-blank.jpg"></img>
          <span className="user-menu_username">{"Anonymous"}</span>
          <span className="connection_status">
            <button className="circular ui icon green button"></button>
          </span>
        </div>
      </div>
    );
  }
}

Navbar.propTypes =  {
  channel: React.PropTypes.string.isRequired
}
